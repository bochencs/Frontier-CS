#include "world.h"
#include <omp.h>
#include <vector>
#include <algorithm>
#include <cmath>

class MySimulator : public Simulator {
private:
    int numThreads;
    int numParticles;

    // Grid parameters
    float cellSize;
    float invCellSize;
    int gridSizeX, gridSizeY;

    // Grid data
    std::vector<int> cellCount;
    std::vector<int> cellStart;
    std::vector<int> cellOfParticle;
    std::vector<int> sortedIndices;

    // Fallback brute-force step (baseline-like)
    void bruteForceStep(std::vector<Particle> &particles,
                        std::vector<Particle> &newParticles,
                        StepParameters params) {
        const int N = (int)particles.size();
        #pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < N; ++i) {
            const Particle pi = particles[i];
            Vec2 force(0.0f, 0.0f);

            for (int j = 0; j < N; ++j) {
                if (j == i) continue;
                Vec2 diff = pi.position - particles[j].position;
                float dist = diff.length();
                if (dist < params.cullRadius) {
                    force += computeForce(pi, particles[j], params.cullRadius);
                }
            }

            newParticles[i] = updateParticle(pi, force, params.deltaTime);
        }
    }

public:
    MySimulator() {
        numThreads = omp_get_max_threads();
        if (numThreads <= 0) numThreads = 8;
        numParticles = 0;
        cellSize = 1.0f;
        invCellSize = 1.0f;
        gridSizeX = gridSizeY = 1;
    }

    void init(int numParticles_, StepParameters params) override {
        numParticles = numParticles_;
        cellSize = params.cullRadius > 0.0f ? params.cullRadius : 1.0f;
        invCellSize = 1.0f / cellSize;

        cellOfParticle.resize(numParticles);
        sortedIndices.resize(numParticles);
        cellCount.clear();
        cellStart.clear();

        omp_set_num_threads(numThreads);
    }

    void simulateStep(std::vector<Particle> &particles,
                      std::vector<Particle> &newParticles,
                      StepParameters params) override {
        const int N = (int)particles.size();
        if (N == 0) return;

        // Extremely small cull radius: fall back to brute-force
        if (params.cullRadius < 1e-3f) {
            bruteForceStep(particles, newParticles, params);
            return;
        }

        cellSize = params.cullRadius;
        invCellSize = 1.0f / cellSize;

        // Compute bounding box (sequential - cheap for N=10k)
        float minX = particles[0].position.x;
        float maxX = minX;
        float minY = particles[0].position.y;
        float maxY = minY;

        for (int i = 1; i < N; ++i) {
            const float x = particles[i].position.x;
            const float y = particles[i].position.y;
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
        }

        const float eps = 1e-3f;
        minX -= eps;
        minY -= eps;
        maxX += eps;
        maxY += eps;

        const float width = maxX - minX;
        const float height = maxY - minY;

        gridSizeX = std::max(1, (int)(width * invCellSize) + 1);
        gridSizeY = std::max(1, (int)(height * invCellSize) + 1);
        const int numCells = gridSizeX * gridSizeY;

        if ((int)cellCount.size() != numCells) {
            cellCount.assign(numCells, 0);
            cellStart.resize(numCells);
        } else {
            std::fill(cellCount.begin(), cellCount.end(), 0);
        }

        if ((int)cellOfParticle.size() != N) {
            cellOfParticle.resize(N);
            sortedIndices.resize(N);
        }

        // Assign particles to cells and count per-cell population
        for (int i = 0; i < N; ++i) {
            const float x = particles[i].position.x;
            const float y = particles[i].position.y;
            int ix = (int)((x - minX) * invCellSize);
            int iy = (int)((y - minY) * invCellSize);
            if (ix < 0) ix = 0;
            else if (ix >= gridSizeX) ix = gridSizeX - 1;
            if (iy < 0) iy = 0;
            else if (iy >= gridSizeY) iy = gridSizeY - 1;

            int cell = iy * gridSizeX + ix;
            cellOfParticle[i] = cell;
            cellCount[cell]++;
        }

        // Prefix sum to get start index of each cell in sortedIndices
        int sum = 0;
        for (int c = 0; c < numCells; ++c) {
            cellStart[c] = sum;
            sum += cellCount[c];
        }

        // Fill sortedIndices grouped by cell (stable within each cell)
        std::fill(cellCount.begin(), cellCount.end(), 0);
        for (int i = 0; i < N; ++i) {
            int cell = cellOfParticle[i];
            int dst = cellStart[cell] + cellCount[cell]++;
            sortedIndices[dst] = i;
        }

        const float cullRadius = params.cullRadius;
        const float cullRadius2 = cullRadius * cullRadius;
        const float deltaTime = params.deltaTime;

        // Because cellSize == cullRadius, we only need to inspect 3x3 neighborhood
        const int neighborRange = 1;

        // Main parallel force computation
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            const Particle pi = particles[i];
            Vec2 force(0.0f, 0.0f);

            int cell = cellOfParticle[i];
            int cx = cell % gridSizeX;
            int cy = cell / gridSizeX;

            for (int dy = -neighborRange; dy <= neighborRange; ++dy) {
                int ny = cy + dy;
                if (ny < 0 || ny >= gridSizeY) continue;

                for (int dx = -neighborRange; dx <= neighborRange; ++dx) {
                    int nx = cx + dx;
                    if (nx < 0 || nx >= gridSizeX) continue;

                    int nCell = ny * gridSizeX + nx;
                    int start = cellStart[nCell];
                    int end = (nCell + 1 < numCells) ? cellStart[nCell + 1] : N;

                    for (int idx = start; idx < end; ++idx) {
                        int j = sortedIndices[idx];
                        if (j == i) continue;

                        const Particle &pj = particles[j];
                        float dxp = pj.position.x - pi.position.x;
                        float dyp = pj.position.y - pi.position.y;
                        float dist2 = dxp * dxp + dyp * dyp;

                        if (dist2 < cullRadius2) {
                            force += computeForce(pi, pj, cullRadius);
                        }
                    }
                }
            }

            newParticles[i] = updateParticle(pi, force, deltaTime);
        }
    }
};

Simulator* createSimulator() {
    return new MySimulator();
}