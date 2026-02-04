#include "world.h"
#include <omp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

class GridSimulator : public Simulator {
private:
    int numThreads = 16;
    
    struct ParticleData {
        int originalIndex;
        Particle p;
    };

    // persistent buffers to minimize allocation overhead
    std::vector<ParticleData> sortedParticles;
    std::vector<int> cellStart;
    std::vector<int> cellEnd;
    std::vector<int> counts;
    std::vector<int> pCellIdx;

public:
    void init(int numParticles, StepParameters params) override {
        numThreads = omp_get_max_threads();
        if (numThreads == 0) numThreads = 1;
        omp_set_num_threads(numThreads);
        
        // Pre-allocate memory
        sortedParticles.reserve(numParticles);
        pCellIdx.reserve(numParticles);
        
        // Estimate grid size - conservative reserve
        // Assuming reasonably dense grid, e.g., 50x50 = 2500 cells
        cellStart.reserve(5000);
        cellEnd.reserve(5000);
        counts.reserve(5000);
    }

    void simulateStep(std::vector<Particle> &particles,
                      std::vector<Particle> &newParticles,
                      StepParameters params) override {
        
        int N = (int)particles.size();
        if (N == 0) return;

        // 1. Compute Bounding Box
        float minX = std::numeric_limits<float>::max();
        float maxX = std::numeric_limits<float>::lowest();
        float minY = std::numeric_limits<float>::max();
        float maxY = std::numeric_limits<float>::lowest();

        #pragma omp parallel for reduction(min:minX, minY) reduction(max:maxX, maxY)
        for (int i = 0; i < N; i++) {
            const Vec2& p = particles[i].position;
            if (p.x < minX) minX = p.x;
            if (p.x > maxX) maxX = p.x;
            if (p.y < minY) minY = p.y;
            if (p.y > maxY) maxY = p.y;
        }

        // Add small padding to avoid boundary conditions
        minX -= 0.1f; minY -= 0.1f;
        maxX += 0.1f; maxY += 0.1f;

        // 2. Setup Grid Parameters
        // Using cell size = cullRadius / 2 provides a good balance between
        // culling efficiency and cell traversal overhead.
        float r = params.cullRadius;
        float cellSize = r * 0.5f;
        // Avoid degeneracy
        if (cellSize < 0.1f) cellSize = 0.1f;
        
        float invCellSize = 1.0f / cellSize;
        int gridW = (int)((maxX - minX) * invCellSize) + 1;
        int gridH = (int)((maxY - minY) * invCellSize) + 1;
        int numCells = gridW * gridH;

        // 3. Prepare Grid Buffers
        if ((size_t)numCells > cellStart.size()) {
            cellStart.resize(numCells);
            cellEnd.resize(numCells);
            counts.resize(numCells);
        }
        
        // Clear counts
        std::fill(counts.begin(), counts.begin() + numCells, 0);

        // 4. Binning (Counting Sort)
        if ((size_t)N > pCellIdx.size()) pCellIdx.resize(N);
        if ((size_t)N > sortedParticles.size()) sortedParticles.resize(N);

        // 4a. Count particles per cell
        for (int i = 0; i < N; i++) {
            const Vec2& p = particles[i].position;
            int cx = (int)((p.x - minX) * invCellSize);
            int cy = (int)((p.y - minY) * invCellSize);
            // Clamp to grid bounds
            if (cx < 0) cx = 0; else if (cx >= gridW) cx = gridW - 1;
            if (cy < 0) cy = 0; else if (cy >= gridH) cy = gridH - 1;
            
            int cIdx = cy * gridW + cx;
            pCellIdx[i] = cIdx;
            counts[cIdx]++;
        }

        // 4b. Prefix Sum (calculate cell offsets)
        int current = 0;
        for (int c = 0; c < numCells; c++) {
            cellStart[c] = current;
            current += counts[c];
            cellEnd[c] = current;
            counts[c] = cellStart[c]; // Reset to use as write pointers
        }

        // 4c. Reorder particles
        for (int i = 0; i < N; i++) {
            int c = pCellIdx[i];
            int dest = counts[c]++;
            sortedParticles[dest].originalIndex = i;
            sortedParticles[dest].p = particles[i];
        }

        // 5. Parallel Force Computation
        float r2 = r * r;
        int searchRad = 2; // Since cellSize = R/2, we search +/- 2 cells

        #pragma omp parallel for schedule(dynamic, 32)
        for (int i = 0; i < N; i++) {
            const Particle& pi = sortedParticles[i].p;
            Vec2 force = {0.0f, 0.0f};

            int cx = (int)((pi.position.x - minX) * invCellSize);
            int cy = (int)((pi.position.y - minY) * invCellSize);
            if (cx < 0) cx = 0; else if (cx >= gridW) cx = gridW - 1;
            if (cy < 0) cy = 0; else if (cy >= gridH) cy = gridH - 1;

            int minCx = std::max(0, cx - searchRad);
            int maxCx = std::min(gridW - 1, cx + searchRad);
            int minCy = std::max(0, cy - searchRad);
            int maxCy = std::min(gridH - 1, cy + searchRad);

            // Iterate over neighbor cells
            for (int ny = minCy; ny <= maxCy; ny++) {
                int yOffset = ny * gridW;
                for (int nx = minCx; nx <= maxCx; nx++) {
                    int cellIdx = yOffset + nx;
                    int start = cellStart[cellIdx];
                    int end = cellEnd[cellIdx];

                    // Check all particles in the neighbor cell
                    for (int k = start; k < end; k++) {
                        // Skip self (same index in sorted array)
                        if (i == k) continue;

                        const Particle& pj = sortedParticles[k].p;
                        Vec2 d = pj.position - pi.position;
                        float dist2 = d.x*d.x + d.y*d.y;
                        
                        // Optimized distance check avoiding sqrt
                        if (dist2 < r2) {
                            force += computeForce(pi, pj, r);
                        }
                    }
                }
            }

            // Write result to the original index
            int origIdx = sortedParticles[i].originalIndex;
            newParticles[origIdx] = updateParticle(pi, force, params.deltaTime);
        }
    }
};

Simulator* createSimulator() {
    return new GridSimulator();
}