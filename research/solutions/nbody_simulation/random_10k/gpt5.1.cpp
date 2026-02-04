#include "world.h"
#include <omp.h>
#include <vector>
#include <cmath>
#include <algorithm>

class MySimulator : public Simulator {
private:
    int numThreads;
    std::vector<int> cellCounts;
    std::vector<int> cellStart;
    std::vector<int> particleCellIndex;
    std::vector<int> sortedParticleIndices;

public:
    MySimulator() : numThreads(omp_get_max_threads()) {}

    void init(int numParticles, StepParameters params) override {
        numThreads = omp_get_max_threads();
        omp_set_num_threads(numThreads);

        particleCellIndex.assign(numParticles, 0);
        sortedParticleIndices.assign(numParticles, 0);
        cellCounts.clear();
        cellStart.clear();
    }

    void simulateStep(std::vector<Particle> &particles,
                      std::vector<Particle> &newParticles,
                      StepParameters params) override {
        const int N = static_cast<int>(particles.size());
        if (N == 0) return;

        if (static_cast<int>(newParticles.size()) < N) {
            newParticles.resize(N);
        }

        const float dt = params.deltaTime;
        const float cullRadius = params.cullRadius;

        // If cullRadius is non-positive, no forces are applied
        if (cullRadius <= 0.0f) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < N; ++i) {
                newParticles[i] = updateParticle(particles[i], Vec2(0.0f, 0.0f), dt);
            }
            return;
        }

        // Compute bounding box of all particles
        float minX = particles[0].position.x;
        float maxX = particles[0].position.x;
        float minY = particles[0].position.y;
        float maxY = particles[0].position.y;

        for (int i = 1; i < N; ++i) {
            const Vec2 &pos = particles[i].position;
            if (pos.x < minX) minX = pos.x;
            if (pos.x > maxX) maxX = pos.x;
            if (pos.y < minY) minY = pos.y;
            if (pos.y > maxY) maxY = pos.y;
        }

        float spanX = maxX - minX;
        float spanY = maxY - minY;
        if (spanX < 1e-3f) spanX = 1e-3f;
        if (spanY < 1e-3f) spanY = 1e-3f;

        // Uniform grid setup
        float cellSize = cullRadius * 0.5f;
        if (cellSize < 1e-3f) cellSize = 1e-3f;
        const float invCellSize = 1.0f / cellSize;

        int gridWidth  = static_cast<int>(spanX * invCellSize) + 1;
        int gridHeight = static_cast<int>(spanY * invCellSize) + 1;
        if (gridWidth < 1) gridWidth = 1;
        if (gridHeight < 1) gridHeight = 1;

        const int numCells = gridWidth * gridHeight;

        cellCounts.assign(numCells, 0);
        cellStart.resize(numCells + 1);

        if (static_cast<int>(particleCellIndex.size()) < N)
            particleCellIndex.resize(N);
        if (static_cast<int>(sortedParticleIndices.size()) < N)
            sortedParticleIndices.resize(N);

        // Assign particles to cells and count particles per cell
        for (int i = 0; i < N; ++i) {
            const Vec2 &pos = particles[i].position;
            int cx = static_cast<int>((pos.x - minX) * invCellSize);
            int cy = static_cast<int>((pos.y - minY) * invCellSize);

            if (cx < 0) cx = 0;
            else if (cx >= gridWidth) cx = gridWidth - 1;
            if (cy < 0) cy = 0;
            else if (cy >= gridHeight) cy = gridHeight - 1;

            int cell = cy * gridWidth + cx;
            particleCellIndex[i] = cell;
            ++cellCounts[cell];
        }

        // Prefix sum: compute start index of each cell in the sorted index array
        cellStart[0] = 0;
        for (int c = 0; c < numCells; ++c) {
            cellStart[c + 1] = cellStart[c] + cellCounts[c];
        }

        // Reuse cellCounts as current write positions
        std::copy(cellStart.begin(), cellStart.begin() + numCells, cellCounts.begin());

        // Fill sortedParticleIndices by cell
        for (int i = 0; i < N; ++i) {
            int cell = particleCellIndex[i];
            int index = cellCounts[cell]++;
            sortedParticleIndices[index] = i;
        }

        const float r2 = cullRadius * cullRadius;
        const int range = static_cast<int>(std::ceil(cullRadius * invCellSize));
        const int gW = gridWidth;
        const int gH = gridHeight;

        // Parallel force computation with spatial culling
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            const Particle &pi = particles[i];
            Vec2 totalForce(0.0f, 0.0f);

            int cell = particleCellIndex[i];
            int cy = cell / gW;
            int cx = cell - cy * gW;

            for (int dy = -range; dy <= range; ++dy) {
                int ny = cy + dy;
                if (ny < 0 || ny >= gH) continue;

                for (int dx = -range; dx <= range; ++dx) {
                    int nx = cx + dx;
                    if (nx < 0 || nx >= gW) continue;

                    int c = ny * gW + nx;
                    int start = cellStart[c];
                    int end   = cellStart[c + 1];

                    for (int idx = start; idx < end; ++idx) {
                        int j = sortedParticleIndices[idx];
                        if (j == i) continue;

                        const Vec2 &pjPos = particles[j].position;
                        float dxp = pjPos.x - pi.position.x;
                        float dyp = pjPos.y - pi.position.y;
                        float dist2 = dxp * dxp + dyp * dyp;

                        // Filter by squared distance to avoid expensive sqrt where unnecessary
                        if (dist2 < r2 && dist2 >= 1e-6f) {
                            totalForce += computeForce(pi, particles[j], cullRadius);
                        }
                    }
                }
            }

            newParticles[i] = updateParticle(pi, totalForce, dt);
        }
    }
};

// Factory function - must be implemented
Simulator* createSimulator() {
    return new MySimulator();
}