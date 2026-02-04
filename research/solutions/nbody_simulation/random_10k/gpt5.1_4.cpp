#include "world.h"
#include <omp.h>
#include <vector>
#include <algorithm>
#include <cmath>

class MySimulator : public Simulator {
private:
    int numThreads = 0;

    // Persistent buffers to reduce allocations
    std::vector<int> cellCounts;      // also reused as cellOffsets
    std::vector<int> cellStart;       // prefix sums, size = numCells + 1
    std::vector<int> sortedIndices;   // particle indices sorted by cell
    std::vector<int> particleCell;    // cell index per particle

    // Brute-force fallback (optimized with squared-distance precheck)
    void simulateBruteForce(const std::vector<Particle> &particles,
                            std::vector<Particle> &newParticles,
                            const StepParameters &params) {
        const int N = (int)particles.size();
        if ((int)newParticles.size() < N) newParticles.resize(N);

        const float cullR = params.cullRadius;
        const float cullR2 = cullR * cullR;
        const float dt = params.deltaTime;

        const Particle *pArr = particles.data();

        #pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < N; ++i) {
            const Particle &pi = pArr[i];
            Vec2 force(0.0f, 0.0f);

            for (int j = 0; j < N; ++j) {
                if (j == i) continue;
                const Particle &pj = pArr[j];

                float dx = pj.position.x - pi.position.x;
                float dy = pj.position.y - pi.position.y;
                float dist2 = dx * dx + dy * dy;

                if (dist2 < cullR2) {
                    force += computeForce(pi, pj, cullR);
                }
            }

            newParticles[i] = updateParticle(pi, force, dt);
        }
    }

public:
    void init(int numParticles, StepParameters params) override {
        numThreads = omp_get_max_threads();
        if (numThreads <= 0) numThreads = 1;
        omp_set_num_threads(numThreads);

        sortedIndices.resize(numParticles);
        particleCell.resize(numParticles);
    }

    void simulateStep(std::vector<Particle> &particles,
                      std::vector<Particle> &newParticles,
                      StepParameters params) override {
        const int N = (int)particles.size();
        if (N == 0) return;
        if ((int)newParticles.size() < N) newParticles.resize(N);

        const float cullR = params.cullRadius;
        if (cullR <= 0.0f) {
            // Degenerate case: no interactions
            const float dt = params.deltaTime;
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < N; ++i) {
                newParticles[i] = updateParticle(particles[i], Vec2(0.0f, 0.0f), dt);
            }
            return;
        }

        const float cullR2 = cullR * cullR;
        const float dt = params.deltaTime;

        // Compute bounding box of current particles
        float minX = particles[0].position.x;
        float maxX = minX;
        float minY = particles[0].position.y;
        float maxY = minY;

        for (int i = 1; i < N; ++i) {
            const Vec2 &pos = particles[i].position;
            if (pos.x < minX) minX = pos.x;
            if (pos.x > maxX) maxX = pos.x;
            if (pos.y < minY) minY = pos.y;
            if (pos.y > maxY) maxY = pos.y;
        }

        float rangeX = maxX - minX;
        float rangeY = maxY - minY;
        const float eps = 1e-3f;
        if (rangeX < eps) rangeX = eps;
        if (rangeY < eps) rangeY = eps;

        const float cellSize = cullR;
        const float invCellSize = 1.0f / cellSize;

        int Nx = (int)(rangeX * invCellSize) + 1;
        int Ny = (int)(rangeY * invCellSize) + 1;

        if (Nx < 1) Nx = 1;
        if (Ny < 1) Ny = 1;

        long long numCellsLL = (long long)Nx * (long long)Ny;
        const long long maxCells = 200000;  // safety bound for pathological cases
        if (numCellsLL <= 0 || numCellsLL > maxCells) {
            // Fallback to brute-force if grid would be too large
            simulateBruteForce(particles, newParticles, params);
            return;
        }

        const int numCells = (int)numCellsLL;

        // Ensure buffers are large enough
        if ((int)cellCounts.size() < numCells) {
            cellCounts.assign(numCells, 0);
        } else {
            std::fill(cellCounts.begin(), cellCounts.begin() + numCells, 0);
        }

        if ((int)cellStart.size() < numCells + 1) {
            cellStart.resize(numCells + 1);
        }

        if ((int)sortedIndices.size() < N) {
            sortedIndices.resize(N);
        }
        if ((int)particleCell.size() < N) {
            particleCell.resize(N);
        }

        // Assign particles to cells and count per-cell population
        for (int i = 0; i < N; ++i) {
            const Vec2 &pos = particles[i].position;
            int cx = (int)((pos.x - minX) * invCellSize);
            int cy = (int)((pos.y - minY) * invCellSize);

            if (cx < 0) cx = 0;
            else if (cx >= Nx) cx = Nx - 1;
            if (cy < 0) cy = 0;
            else if (cy >= Ny) cy = Ny - 1;

            int cellIdx = cy * Nx + cx;
            particleCell[i] = cellIdx;
            cellCounts[cellIdx]++;
        }

        // Prefix sum to get starting indices for each cell
        cellStart[0] = 0;
        for (int c = 0; c < numCells; ++c) {
            cellStart[c + 1] = cellStart[c] + cellCounts[c];
        }

        // Use cellCounts as temporary offsets (initialized from cellStart)
        for (int c = 0; c < numCells; ++c) {
            cellCounts[c] = cellStart[c];
        }

        // Fill sortedIndices with particle indices grouped by cell
        for (int i = 0; i < N; ++i) {
            int cellIdx = particleCell[i];
            int dst = cellCounts[cellIdx]++;
            sortedIndices[dst] = i;
        }

        const Particle *pArr = particles.data();

        // Parallel force computation using uniform grid neighborhood search
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            const Particle &pi = pArr[i];
            Vec2 force(0.0f, 0.0f);

            int cellIdx = particleCell[i];
            int cy = cellIdx / Nx;
            int cx = cellIdx - cy * Nx;

            for (int dy = -1; dy <= 1; ++dy) {
                int ny = cy + dy;
                if (ny < 0 || ny >= Ny) continue;
                int yOffset = ny * Nx;

                for (int dx = -1; dx <= 1; ++dx) {
                    int nxCell = cx + dx;
                    if (nxCell < 0 || nxCell >= Nx) continue;

                    int neighborCell = yOffset + nxCell;
                    int start = cellStart[neighborCell];
                    int end   = cellStart[neighborCell + 1];

                    for (int idx = start; idx < end; ++idx) {
                        int j = sortedIndices[idx];
                        if (j == i) continue;

                        const Particle &pj = pArr[j];
                        float dxp = pj.position.x - pi.position.x;
                        float dyp = pj.position.y - pi.position.y;
                        float dist2 = dxp * dxp + dyp * dyp;

                        if (dist2 < cullR2) {
                            force += computeForce(pi, pj, cullR);
                        }
                    }
                }
            }

            newParticles[i] = updateParticle(pi, force, dt);
        }
    }
};

// Factory function - must be implemented
Simulator* createSimulator() {
    return new MySimulator();
}