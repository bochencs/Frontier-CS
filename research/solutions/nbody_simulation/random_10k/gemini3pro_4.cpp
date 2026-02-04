#include "world.h"
#include <omp.h>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>

class MySimulator : public Simulator {
private:
    int numThreads;

    // Persistent buffers to avoid reallocation across steps
    // sortedParticles stores particles reordered by grid cell for cache locality
    std::vector<Particle> sortedParticles;
    // originalIndices maps sorted index back to original index for output
    std::vector<int> originalIndices;
    // particleCellIndices stores the grid cell index for each particle
    std::vector<int> particleCellIndices;

    // Grid data structures
    // gridCounts: number of particles in each cell
    std::vector<int> gridCounts;
    // gridOffsets: starting index of each cell in sortedParticles
    std::vector<int> gridOffsets;
    // currentPos: temporary buffer used during sorting
    std::vector<int> currentPos;

public:
    MySimulator() {
        // Use maximum available threads for parallel sections
        numThreads = omp_get_max_threads();
    }

    void init(int numParticles, StepParameters params) override {
        omp_set_num_threads(numThreads);
        
        // Reserve memory for particle-sized buffers
        if (sortedParticles.capacity() < (size_t)numParticles) {
            sortedParticles.reserve(numParticles);
            originalIndices.reserve(numParticles);
            particleCellIndices.reserve(numParticles);
        }
    }

    void simulateStep(std::vector<Particle> &particles,
                      std::vector<Particle> &newParticles,
                      StepParameters params) override {
        
        int N = (int)particles.size();
        if (N == 0) return;

        // Ensure buffers are sized correctly (fast if size matches)
        if (sortedParticles.size() != (size_t)N) {
            sortedParticles.resize(N);
            originalIndices.resize(N);
            particleCellIndices.resize(N);
        }

        // 1. Calculate Bounding Box of the world to determine grid dimensions
        float minX = std::numeric_limits<float>::max();
        float maxX = std::numeric_limits<float>::lowest();
        float minY = std::numeric_limits<float>::max();
        float maxY = std::numeric_limits<float>::lowest();

        // Parallel reduction is efficient for finding min/max
        #pragma omp parallel for reduction(min:minX, minY) reduction(max:maxX, maxY)
        for (int i = 0; i < N; i++) {
            const auto& p = particles[i].position;
            if (p.x < minX) minX = p.x;
            if (p.x > maxX) maxX = p.x;
            if (p.y < minY) minY = p.y;
            if (p.y > maxY) maxY = p.y;
        }

        // 2. Setup Grid Dimensions
        // Use cullRadius as cell size to optimize neighbor search (3x3 blocks)
        float cellSize = params.cullRadius;
        if (cellSize <= 1e-4f) cellSize = 1e-4f; // Safety against zero radius

        int gridW = static_cast<int>((maxX - minX) / cellSize) + 1;
        int gridH = static_cast<int>((maxY - minY) / cellSize) + 1;
        int numCells = gridW * gridH;

        // Resize grid buffers if necessary
        if ((int)gridCounts.size() < numCells) {
            gridCounts.resize(numCells);
            gridOffsets.resize(numCells);
            currentPos.resize(numCells);
        }
        
        // Reset grid counts
        // Using std::fill is fast; only clear the used portion
        std::fill(gridCounts.begin(), gridCounts.begin() + numCells, 0);

        // 3. Assign Particles to Cells (Binning)
        // Serial execution is sufficiently fast for 10k particles and avoids atomic overhead
        for (int i = 0; i < N; i++) {
            int cx = static_cast<int>((particles[i].position.x - minX) / cellSize);
            int cy = static_cast<int>((particles[i].position.y - minY) / cellSize);
            
            // Clamp to grid bounds to handle floating point inaccuracies
            if (cx < 0) cx = 0; else if (cx >= gridW) cx = gridW - 1;
            if (cy < 0) cy = 0; else if (cy >= gridH) cy = gridH - 1;
            
            int cellIdx = cy * gridW + cx;
            particleCellIndices[i] = cellIdx;
            gridCounts[cellIdx]++;
        }

        // 4. Compute Offsets (Prefix Sum)
        // Determines where each cell's particles start in the sorted array
        int accum = 0;
        for (int c = 0; c < numCells; c++) {
            gridOffsets[c] = accum;
            currentPos[c] = accum; // Initialize currentPos for the sorting phase
            accum += gridCounts[c];
        }

        // 5. Sort Particles by Cell
        // Reordering particles improves data locality during force calculation
        for (int i = 0; i < N; i++) {
            int cellIdx = particleCellIndices[i];
            int destIdx = currentPos[cellIdx]++;
            sortedParticles[destIdx] = particles[i];
            originalIndices[destIdx] = i; // Keep track of original index to write back results correctly
        }

        // 6. Compute Forces in Parallel
        float cullRadiusSq = params.cullRadius * params.cullRadius;

        // Dynamic scheduling helps load balance if density varies
        #pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < N; i++) {
            const Particle& pi = sortedParticles[i];
            Vec2 force = {0.0f, 0.0f};

            // Recalculate cell coords for pi (cheap)
            int cx = static_cast<int>((pi.position.x - minX) / cellSize);
            int cy = static_cast<int>((pi.position.y - minY) / cellSize);
            
            // Clamp
            if (cx < 0) cx = 0; else if (cx >= gridW) cx = gridW - 1;
            if (cy < 0) cy = 0; else if (cy >= gridH) cy = gridH - 1;

            // Iterate over 3x3 neighbor cells
            int minNx = std::max(0, cx - 1);
            int maxNx = std::min(gridW - 1, cx + 1);
            int minNy = std::max(0, cy - 1);
            int maxNy = std::min(gridH - 1, cy + 1);

            for (int ny = minNy; ny <= maxNy; ny++) {
                int yOffset = ny * gridW;
                for (int nx = minNx; nx <= maxNx; nx++) {
                    int nCellIdx = yOffset + nx;
                    
                    int start = gridOffsets[nCellIdx];
                    int end = start + gridCounts[nCellIdx];

                    // Iterate over particles in neighbor cell
                    // Accessing sortedParticles linearly is cache-friendly
                    for (int j = start; j < end; j++) {
                        // Avoid self-interaction
                        if (i == j) continue;

                        const Particle& pj = sortedParticles[j];
                        
                        // Optimization: Check squared distance before calling computeForce
                        // This avoids sqrt and function call overhead for distant particles
                        Vec2 d = pj.position - pi.position;
                        float distSq = d.x*d.x + d.y*d.y;

                        if (distSq < cullRadiusSq) {
                            force += computeForce(pi, pj, params.cullRadius);
                        }
                    }
                }
            }

            // Update particle and write to newParticles at the original index
            newParticles[originalIndices[i]] = updateParticle(pi, force, params.deltaTime);
        }
    }
};

Simulator* createSimulator() {
    return new MySimulator();
}