#include "world.h"
#include <omp.h>
#include <vector>
#include <cmath>
#include <algorithm>

class MySimulator : public Simulator {
private:
    int numThreads = 0;

    // Grid related persistent buffers
    std::vector<int> cellCounts;
    std::vector<int> cellStarts;
    std::vector<int> cellOffsets;
    std::vector<int> sortedIndices;
    std::vector<int> particleCells;

public:
    void init(int numParticles, StepParameters params) override {
        numThreads = omp_get_max_threads();
        if (numThreads <= 0) numThreads = 1;
        omp_set_num_threads(numThreads);

        sortedIndices.resize(numParticles);
        particleCells.resize(numParticles);
        cellCounts.clear();
        cellStarts.clear();
        cellOffsets.clear();
    }

    void simulateStep(std::vector<Particle> &particles,
                      std::vector<Particle> &newParticles,
                      StepParameters params) override {
        int N = (int)particles.size();
        if (N == 0) return;

        if ((int)sortedIndices.size() < N) {
            sortedIndices.resize(N);
            particleCells.resize(N);
        }

        float dt = params.deltaTime;
        float r = params.cullRadius;
        float r2 = r * r;

        // If cullRadius <= 0, no forces act
        if (r <= 0.0f) {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < N; ++i) {
                newParticles[i] = updateParticle(particles[i], Vec2(0.0f, 0.0f), dt);
            }
            return;
        }

        // Compute bounding box of current particle positions
        float minX = particles[0].position.x;
        float maxX = minX;
        float minY = particles[0].position.y;
        float maxY = minY;

        for (int i = 1; i < N; ++i) {
            float x = particles[i].position.x;
            float y = particles[i].position.y;
            if (x < minX) minX = x;
            else if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            else if (y > maxY) maxY = y;
        }

        // Slight padding to be robust against floating point rounding at boundaries
        const float eps = 1e-3f;
        minX -= eps;
        minY -= eps;
        maxX += eps;
        maxY += eps;

        // Grid cell size: use cullRadius (good tradeoff for given problem)
        float cellSize = r;
        if (cellSize < 1e-3f) cellSize = 1e-3f;
        float invCellSize = 1.0f / cellSize;

        float extentX = maxX - minX;
        float extentY = maxY - minY;

        int numCellsX = (int)(extentX * invCellSize) + 1;
        int numCellsY = (int)(extentY * invCellSize) + 1;
        if (numCellsX < 1) numCellsX = 1;
        if (numCellsY < 1) numCellsY = 1;

        size_t numCells = (size_t)numCellsX * (size_t)numCellsY;

        // Safety cap on number of cells; fall back to brute-force if too many
        const size_t MAX_CELLS = 1000000; // 1M cells upper bound
        if (numCells > MAX_CELLS) {
            // Brute-force with distance-squared culling (still better than baseline)
#pragma omp parallel for schedule(dynamic, 16)
            for (int i = 0; i < N; ++i) {
                const Particle &pi = particles[i];
                float px = pi.position.x;
                float py = pi.position.y;
                Vec2 force(0.0f, 0.0f);

                for (int j = 0; j < N; ++j) {
                    if (j == i) continue;
                    const Particle &pj = particles[j];
                    float dx = pj.position.x - px;
                    float dy = pj.position.y - py;
                    float dist2 = dx * dx + dy * dy;
                    if (dist2 <= r2) {
                        force += computeForce(pi, pj, r);
                    }
                }
                newParticles[i] = updateParticle(pi, force, dt);
            }
            return;
        }

        // Prepare / resize grid buffers
        if (cellCounts.size() < numCells) {
            cellCounts.assign(numCells, 0);
            cellStarts.assign(numCells, 0);
            cellOffsets.assign(numCells, 0);
        } else {
            std::fill(cellCounts.begin(), cellCounts.begin() + numCells, 0);
        }

        // Assign particles to cells and count
        for (int i = 0; i < N; ++i) {
            float x = particles[i].position.x;
            float y = particles[i].position.y;

            int cx = (int)((x - minX) * invCellSize);
            int cy = (int)((y - minY) * invCellSize);

            if (cx < 0) cx = 0;
            else if (cx >= numCellsX) cx = numCellsX - 1;
            if (cy < 0) cy = 0;
            else if (cy >= numCellsY) cy = numCellsY - 1;

            int cellIndex = cy * numCellsX + cx;
            particleCells[i] = cellIndex;
            cellCounts[cellIndex]++;
        }

        // Prefix sum to get starts and working offsets
        int sum = 0;
        for (size_t c = 0; c < numCells; ++c) {
            cellStarts[c] = sum;
            cellOffsets[c] = sum;
            sum += cellCounts[c];
        }

        // Fill sortedIndices so that particles of each cell are contiguous
        for (int i = 0; i < N; ++i) {
            int cellIndex = particleCells[i];
            int dest = cellOffsets[cellIndex]++;
            sortedIndices[dest] = i;
        }

        // Main force computation using grid-based neighbor search
#pragma omp parallel for schedule(dynamic, 32)
        for (int i = 0; i < N; ++i) {
            const Particle &pi = particles[i];
            float px = pi.position.x;
            float py = pi.position.y;

            // Spatial search bounds
            float minSearchX = px - r;
            float maxSearchX = px + r;
            float minSearchY = py - r;
            float maxSearchY = py + r;

            int minCx = (int)std::floor((minSearchX - minX) * invCellSize);
            int maxCx = (int)std::floor((maxSearchX - minX) * invCellSize);
            int minCy = (int)std::floor((minSearchY - minY) * invCellSize);
            int maxCy = (int)std::floor((maxSearchY - minY) * invCellSize);

            if (minCx < 0) minCx = 0;
            if (maxCx >= numCellsX) maxCx = numCellsX - 1;
            if (minCy < 0) minCy = 0;
            if (maxCy >= numCellsY) maxCy = numCellsY - 1;

            Vec2 force(0.0f, 0.0f);

            for (int cy = minCy; cy <= maxCy; ++cy) {
                int rowBase = cy * numCellsX;
                for (int cx = minCx; cx <= maxCx; ++cx) {
                    int cellIndex = rowBase + cx;
                    int start = cellStarts[cellIndex];
                    int count = cellCounts[cellIndex];
                    int end = start + count;

                    for (int idx = start; idx < end; ++idx) {
                        int j = sortedIndices[idx];
                        if (j == i) continue;

                        const Particle &pj = particles[j];
                        float dx = pj.position.x - px;
                        float dy = pj.position.y - py;
                        float d2 = dx * dx + dy * dy;
                        if (d2 <= r2) {
                            force += computeForce(pi, pj, r);
                        }
                    }
                }
            }

            newParticles[i] = updateParticle(pi, force, dt);
        }
    }
};

Simulator* createSimulator() {
    return new MySimulator();
}