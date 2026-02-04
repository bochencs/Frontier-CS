#include "world.h"
#include <omp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

class OptimizedSimulator : public Simulator {
private:
    int numThreads = 16;
    
    // Persistent buffers to minimize allocation overhead
    std::vector<int> sortedIndices;
    std::vector<int> cellStarts;
    std::vector<int> pCell;
    std::vector<int> cellCounts;
    std::vector<int> currentOffsets;

public:
    void init(int numParticles, StepParameters params) override {
        numThreads = omp_get_max_threads();
        if (numThreads < 1) numThreads = 1;
        omp_set_num_threads(numThreads);
        
        // Reserve memory for grid structures
        // Vectors will be resized in simulateStep if grid dimensions change,
        // but reserving for N avoids frequent particle-buffer reallocs.
        sortedIndices.reserve(numParticles);
        pCell.reserve(numParticles);
        cellCounts.reserve(numParticles);
        cellStarts.reserve(numParticles);
        currentOffsets.reserve(numParticles);
    }

    void simulateStep(std::vector<Particle> &particles,
                      std::vector<Particle> &newParticles,
                      StepParameters params) override {
        
        int N = (int)particles.size();
        if (N == 0) return;

        // 1. Determine World Bounds
        // A simple serial scan is efficient enough for 10k particles
        float minX = std::numeric_limits<float>::max();
        float maxX = std::numeric_limits<float>::lowest();
        float minY = std::numeric_limits<float>::max();
        float maxY = std::numeric_limits<float>::lowest();

        for (const auto& p : particles) {
            if (p.position.x < minX) minX = p.position.x;
            if (p.position.x > maxX) maxX = p.position.x;
            if (p.position.y < minY) minY = p.position.y;
            if (p.position.y > maxY) maxY = p.position.y;
        }

        // Add padding to ensure particles at max boundary fall into valid cells
        const float padding = 0.1f;
        minX -= padding; maxX += padding;
        minY -= padding; maxY += padding;
        
        float width = maxX - minX;
        float height = maxY - minY;

        // 2. Configure Uniform Grid
        // A cell size of cullRadius / 2 provides a good balance between 
        // culling efficiency and grid traversal overhead.
        float cellSize = params.cullRadius * 0.5f;
        // Clamp to avoid degenerate grid with extremely small radius
        if (cellSize < 0.25f) cellSize = 0.25f;

        int gridW = (int)(width / cellSize) + 1;
        int gridH = (int)(height / cellSize) + 1;
        int numCells = gridW * gridH;
        float invCellSize = 1.0f / cellSize;

        // Resize and reset grid buffers
        if ((int)cellCounts.size() < numCells) cellCounts.resize(numCells);
        // We only clear the portion of cellCounts we will use
        std::fill(cellCounts.begin(), cellCounts.begin() + numCells, 0);
        
        if ((int)pCell.size() < N) pCell.resize(N);

        // 3. Bin Particles (Counting Sort Phase 1)
        // Assign particles to cells and count cell population
        for (int i = 0; i < N; i++) {
            int cx = (int)((particles[i].position.x - minX) * invCellSize);
            int cy = (int)((particles[i].position.y - minY) * invCellSize);
            
            // Safety clamping
            if (cx < 0) cx = 0; else if (cx >= gridW) cx = gridW - 1;
            if (cy < 0) cy = 0; else if (cy >= gridH) cy = gridH - 1;
            
            int cellIdx = cy * gridW + cx;
            pCell[i] = cellIdx;
            cellCounts[cellIdx]++;
        }

        // 4. Compute Cell Starts (Prefix Sum)
        if ((int)cellStarts.size() < numCells + 1) cellStarts.resize(numCells + 1);
        int accum = 0;
        for (int i = 0; i < numCells; i++) {
            cellStarts[i] = accum;
            accum += cellCounts[i];
        }
        cellStarts[numCells] = accum;

        // 5. Build Sorted Index List (Counting Sort Phase 2)
        if ((int)sortedIndices.size() < N) sortedIndices.resize(N);
        if ((int)currentOffsets.size() < numCells) currentOffsets.resize(numCells);
        
        // Copy starts to use as write pointers
        std::copy(cellStarts.begin(), cellStarts.begin() + numCells, currentOffsets.begin());

        for (int i = 0; i < N; i++) {
            int c = pCell[i];
            sortedIndices[currentOffsets[c]++] = i;
        }

        // 6. Parallel Force Computation
        float cullSq = params.cullRadius * params.cullRadius;
        float dt = params.deltaTime;
        float radius = params.cullRadius;
        
        // With cellSize = R/2, a radius R covers 2 cells. 
        // We search +/- 3 cells to ensure coverage including boundary conditions.
        const int searchRad = 3;

        #pragma omp parallel for schedule(dynamic, 32)
        for (int i = 0; i < N; i++) {
            const Particle& pi = particles[i];
            
            // Recompute particle's grid coordinates
            int cx = (int)((pi.position.x - minX) * invCellSize);
            int cy = (int)((pi.position.y - minY) * invCellSize);
            if (cx < 0) cx = 0; else if (cx >= gridW) cx = gridW - 1;
            if (cy < 0) cy = 0; else if (cy >= gridH) cy = gridH - 1;

            Vec2 force = Vec2(0.0f, 0.0f);

            // Determine grid search range
            int xMin = std::max(0, cx - searchRad);
            int xMax = std::min(gridW - 1, cx + searchRad);
            int yMin = std::max(0, cy - searchRad);
            int yMax = std::min(gridH - 1, cy + searchRad);

            for (int ny = yMin; ny <= yMax; ny++) {
                // Y-axis AABB Pruning: check if cell is within range of pi.y
                float cellMinY = minY + ny * cellSize;
                float dy = 0.0f;
                if (pi.position.y < cellMinY) dy = cellMinY - pi.position.y;
                else if (pi.position.y > cellMinY + cellSize) dy = pi.position.y - (cellMinY + cellSize);
                
                if (dy > radius) continue;
                float dy2 = dy * dy;

                for (int nx = xMin; nx <= xMax; nx++) {
                    // X-axis AABB Pruning
                    float cellMinX = minX + nx * cellSize;
                    float dx = 0.0f;
                    if (pi.position.x < cellMinX) dx = cellMinX - pi.position.x;
                    else if (pi.position.x > cellMinX + cellSize) dx = pi.position.x - (cellMinX + cellSize);

                    if (dx * dx + dy2 > cullSq) continue;

                    // Iterate over particles in valid neighbor cell
                    int cellIdx = ny * gridW + nx;
                    int start = cellStarts[cellIdx];
                    int end = cellStarts[cellIdx + 1];

                    for (int k = start; k < end; k++) {
                        int j = sortedIndices[k];
                        if (i == j) continue;

                        const Particle& pj = particles[j];
                        
                        // Check distance
                        float pdx = pj.position.x - pi.position.x;
                        float pdy = pj.position.y - pi.position.y;
                        float d2 = pdx*pdx + pdy*pdy;

                        if (d2 < cullSq) {
                            force += computeForce(pi, pj, radius);
                        }
                    }
                }
            }
            newParticles[i] = updateParticle(pi, force, dt);
        }
    }
};

Simulator* createSimulator() {
    return new OptimizedSimulator();
}