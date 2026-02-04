#include "world.h"
#include <omp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>

class MySimulator : public Simulator {
private:
    // Grid-based spatial partitioning
    struct GridCell {
        std::vector<int> particleIndices;
        float minX, maxX, minY, maxY;
        
        GridCell(float x1, float x2, float y1, float y2) 
            : minX(x1), maxX(x2), minY(y1), maxY(y2) {}
        
        void clear() { particleIndices.clear(); }
    };
    
    int numThreads;
    std::vector<GridCell> gridCells;
    std::vector<int> cellAssignment;
    float spaceSize;
    float cellSize;
    int gridDim;
    
    void buildGrid(const std::vector<Particle>& particles, float cullRadius) {
        // Clear all grid cells
        #pragma omp parallel for
        for (size_t i = 0; i < gridCells.size(); i++) {
            gridCells[i].clear();
        }
        
        // Assign particles to grid cells in parallel
        #pragma omp parallel for
        for (int i = 0; i < (int)particles.size(); i++) {
            const Particle& p = particles[i];
            int cellX = std::min((int)(p.position.x / cellSize), gridDim - 1);
            int cellY = std::min((int)(p.position.y / cellSize), gridDim - 1);
            int cellIdx = cellY * gridDim + cellX;
            cellAssignment[i] = cellIdx;
            
            // Use critical section for thread-safe insertion
            #pragma omp critical
            {
                gridCells[cellIdx].particleIndices.push_back(i);
            }
        }
    }
    
    // Check if two cells are within cullRadius
    bool cellsNearby(int cellIdx1, int cellIdx2, float cullRadius) {
        int x1 = cellIdx1 % gridDim;
        int y1 = cellIdx1 / gridDim;
        int x2 = cellIdx2 % gridDim;
        int y2 = cellIdx2 / gridDim;
        
        // Manhattan distance check (conservative but fast)
        int dx = std::abs(x1 - x2);
        int dy = std::abs(y1 - y2);
        return dx <= 2 && dy <= 2; // Check up to 2 cells away in each direction
    }

public:
    MySimulator() : numThreads(16), spaceSize(100.0f) {}
    
    void init(int numParticles, StepParameters params) override {
        numThreads = omp_get_max_threads();
        omp_set_num_threads(numThreads);
        
        // Initialize grid structure - cell size slightly larger than cullRadius
        // to reduce neighbor cell checks
        cellSize = params.cullRadius * 1.5f;
        gridDim = (int)std::ceil(spaceSize / cellSize);
        
        // Create grid cells
        gridCells.reserve(gridDim * gridDim);
        for (int y = 0; y < gridDim; y++) {
            for (int x = 0; x < gridDim; x++) {
                float minX = x * cellSize;
                float maxX = (x + 1) * cellSize;
                float minY = y * cellSize;
                float maxY = (y + 1) * cellSize;
                gridCells.emplace_back(minX, maxX, minY, maxY);
            }
        }
        
        cellAssignment.resize(numParticles);
    }
    
    void simulateStep(std::vector<Particle>& particles,
                      std::vector<Particle>& newParticles,
                      StepParameters params) override {
        // Build spatial grid
        buildGrid(particles, params.cullRadius);
        
        // Parallel force computation with dynamic load balancing
        #pragma omp parallel for schedule(dynamic, 32)
        for (int i = 0; i < (int)particles.size(); i++) {
            const Particle& pi = particles[i];
            Vec2 force = Vec2(0.0f, 0.0f);
            int cellIdx = cellAssignment[i];
            
            // Only check particles in nearby cells
            for (int cy = -2; cy <= 2; cy++) {
                for (int cx = -2; cx <= 2; cx++) {
                    int checkX = (cellIdx % gridDim) + cx;
                    int checkY = (cellIdx / gridDim) + cy;
                    
                    if (checkX >= 0 && checkX < gridDim && checkY >= 0 && checkY < gridDim) {
                        int neighborCellIdx = checkY * gridDim + checkX;
                        
                        // Quick distance check between cell centers
                        float cellCenterX1 = (cellIdx % gridDim) * cellSize + cellSize * 0.5f;
                        float cellCenterY1 = (cellIdx / gridDim) * cellSize + cellSize * 0.5f;
                        float cellCenterX2 = checkX * cellSize + cellSize * 0.5f;
                        float cellCenterY2 = checkY * cellSize + cellSize * 0.5f;
                        float cellDist = std::sqrt(
                            (cellCenterX1 - cellCenterX2) * (cellCenterX1 - cellCenterX2) +
                            (cellCenterY1 - cellCenterY2) * (cellCenterY1 - cellCenterY2)
                        );
                        
                        // Skip cells that are definitely too far
                        if (cellDist > params.cullRadius + cellSize * 1.5f) {
                            continue;
                        }
                        
                        // Check particles in this neighbor cell
                        const auto& neighborIndices = gridCells[neighborCellIdx].particleIndices;
                        for (int j : neighborIndices) {
                            if (j == i) continue;
                            
                            // Early distance check using squared distance
                            Vec2 diff = particles[j].position - pi.position;
                            float dist2 = diff.x * diff.x + diff.y * diff.y;
                            float cullRadius2 = params.cullRadius * params.cullRadius;
                            
                            if (dist2 < cullRadius2 && dist2 > 1e-6f) {
                                force += computeForce(pi, particles[j], params.cullRadius);
                            }
                        }
                    }
                }
            }
            
            newParticles[i] = updateParticle(pi, force, params.deltaTime);
        }
    }
};

Simulator* createSimulator() {
    return new MySimulator();
}