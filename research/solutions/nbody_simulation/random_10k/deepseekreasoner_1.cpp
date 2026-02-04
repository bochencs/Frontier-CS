#include "world.h"
#include <omp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>

class MySimulator : public Simulator {
private:
    int numThreads = 16;
    float cellSize;
    float invCellSize;
    int gridWidth, gridHeight;
    float spaceSize = 100.0f;
    float currentCullRadius;
    
    struct Cell {
        std::vector<int> particleIndices;
        void clear() { particleIndices.clear(); }
    };
    
    std::vector<Cell> grid;
    std::vector<int> cellOffsets;
    std::vector<int> cellCounts;
    
    inline int getCellIndex(float x, float y) const {
        int cx = static_cast<int>(x * invCellSize);
        int cy = static_cast<int>(y * invCellSize);
        cx = std::max(0, std::min(gridWidth - 1, cx));
        cy = std::max(0, std::min(gridHeight - 1, cy));
        return cy * gridWidth + cx;
    }
    
    void rebuildGrid(const std::vector<Particle>& particles, float cullRadius) {
        if (std::abs(currentCullRadius - cullRadius) > 1e-6f) {
            currentCullRadius = cullRadius;
            cellSize = cullRadius;
            invCellSize = 1.0f / cellSize;
            gridWidth = static_cast<int>(std::ceil(spaceSize / cellSize));
            gridHeight = gridWidth;
            grid.resize(gridWidth * gridHeight);
        }
        
        #pragma omp parallel for
        for (size_t i = 0; i < grid.size(); ++i) {
            grid[i].clear();
        }
        
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(particles.size()); ++i) {
            const auto& p = particles[i];
            int cellIdx = getCellIndex(p.position.x, p.position.y);
            #pragma omp critical
            grid[cellIdx].particleIndices.push_back(i);
        }
    }

public:
    void init(int numParticles, StepParameters params) override {
        omp_set_num_threads(numThreads);
        currentCullRadius = params.cullRadius;
        cellSize = params.cullRadius;
        invCellSize = 1.0f / cellSize;
        gridWidth = static_cast<int>(std::ceil(spaceSize / cellSize));
        gridHeight = gridWidth;
        grid.resize(gridWidth * gridHeight);
        cellOffsets.resize(grid.size() + 1);
        cellCounts.resize(grid.size());
    }

    void simulateStep(std::vector<Particle> &particles,
                      std::vector<Particle> &newParticles,
                      StepParameters params) override {
        rebuildGrid(particles, params.cullRadius);
        
        const int gsize = static_cast<int>(grid.size());
        const float cr = params.cullRadius;
        const float cr2 = cr * cr;
        const int maxSearchRadius = 1;
        
        #pragma omp parallel for schedule(dynamic, 32)
        for (int i = 0; i < static_cast<int>(particles.size()); ++i) {
            const Particle& pi = particles[i];
            Vec2 force(0.0f, 0.0f);
            
            int cellX = static_cast<int>(pi.position.x * invCellSize);
            int cellY = static_cast<int>(pi.position.y * invCellSize);
            cellX = std::max(0, std::min(gridWidth - 1, cellX));
            cellY = std::max(0, std::min(gridHeight - 1, cellY));
            
            for (int dy = -maxSearchRadius; dy <= maxSearchRadius; ++dy) {
                int ny = cellY + dy;
                if (ny < 0 || ny >= gridHeight) continue;
                
                for (int dx = -maxSearchRadius; dx <= maxSearchRadius; ++dx) {
                    int nx = cellX + dx;
                    if (nx < 0 || nx >= gridWidth) continue;
                    
                    int cellIdx = ny * gridWidth + nx;
                    const auto& cell = grid[cellIdx];
                    
                    for (int j : cell.particleIndices) {
                        if (j == i) continue;
                        const Particle& pj = particles[j];
                        
                        float dx2 = pi.position.x - pj.position.x;
                        float dy2 = pi.position.y - pj.position.y;
                        float dist2 = dx2 * dx2 + dy2 * dy2;
                        
                        if (dist2 < cr2 && dist2 > 1e-6f) {
                            force += computeForce(pi, pj, cr);
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