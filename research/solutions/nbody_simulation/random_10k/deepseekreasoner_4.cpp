#include "world.h"
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include <array>
#include <cstring>

class MySimulator : public Simulator {
private:
    int numThreads;
    int gridSize;
    float cellSize;
    float invCellSize;
    int totalCells;
    
    struct GridCell {
        std::vector<int> particleIndices;
    };
    
    std::vector<GridCell> grid;
    std::vector<int> cellForParticle;
    std::vector<float> sqrtCache;
    std::vector<int> neighborOffsets;
    
    void buildGrid(const std::vector<Particle>& particles, float cullRadius) {
        cellSize = cullRadius;
        invCellSize = 1.0f / cellSize;
        gridSize = static_cast<int>(100.0f / cellSize) + 1;
        totalCells = gridSize * gridSize;
        
        if (grid.size() != (size_t)totalCells) {
            grid.resize(totalCells);
        }
        
        #pragma omp parallel for
        for (int i = 0; i < totalCells; ++i) {
            grid[i].particleIndices.clear();
        }
        
        #pragma omp parallel for
        for (int i = 0; i < (int)particles.size(); ++i) {
            int cellX = static_cast<int>(particles[i].position.x * invCellSize);
            int cellY = static_cast<int>(particles[i].position.y * invCellSize);
            cellX = std::max(0, std::min(cellX, gridSize - 1));
            cellY = std::max(0, std::min(cellY, gridSize - 1));
            int cellIdx = cellY * gridSize + cellX;
            
            #pragma omp critical
            {
                cellForParticle[i] = cellIdx;
                grid[cellIdx].particleIndices.push_back(i);
            }
        }
    }
    
    void precomputeNeighborOffsets() {
        neighborOffsets.clear();
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                neighborOffsets.push_back(dy * gridSize + dx);
            }
        }
    }

public:
    MySimulator() : numThreads(16) {}
    
    void init(int numParticles, StepParameters params) override {
        omp_set_num_threads(numThreads);
        cellForParticle.resize(numParticles);
        
        if (sqrtCache.empty()) {
            sqrtCache.resize(10000);
            for (int i = 0; i < 10000; ++i) {
                sqrtCache[i] = std::sqrt(static_cast<float>(i));
            }
        }
    }
    
    void simulateStep(std::vector<Particle> &particles,
                      std::vector<Particle> &newParticles,
                      StepParameters params) override {
        
        buildGrid(particles, params.cullRadius);
        
        if (neighborOffsets.empty()) {
            precomputeNeighborOffsets();
        }
        
        const float cullRadiusSq = params.cullRadius * params.cullRadius;
        const float cullRadius75 = params.cullRadius * 0.75f;
        const float decaySlope = 4.0f / params.cullRadius;
        
        #pragma omp parallel for schedule(dynamic, 32)
        for (int i = 0; i < (int)particles.size(); ++i) {
            const Particle& pi = particles[i];
            Vec2 force(0.0f, 0.0f);
            
            int cellIdx = cellForParticle[i];
            int cellY = cellIdx / gridSize;
            int cellX = cellIdx % gridSize;
            
            for (int offset : neighborOffsets) {
                int nCellY = cellY + offset / gridSize;
                int nCellX = cellX + offset % gridSize;
                
                if (nCellX < 0 || nCellX >= gridSize || nCellY < 0 || nCellY >= gridSize) {
                    continue;
                }
                
                int neighborCellIdx = nCellY * gridSize + nCellX;
                const auto& neighborIndices = grid[neighborCellIdx].particleIndices;
                
                for (int j : neighborIndices) {
                    if (j == i) continue;
                    
                    const Particle& pj = particles[j];
                    Vec2 dir = pj.position - pi.position;
                    float distSq = dir.x * dir.x + dir.y * dir.y;
                    
                    if (distSq > cullRadiusSq) continue;
                    if (distSq < 1e-6f) continue;
                    
                    float dist = distSq < 10000.0f ? sqrtCache[static_cast<int>(distSq)] : std::sqrt(distSq);
                    
                    if (dist < 1e-3f) continue;
                    dir.x /= dist;
                    dir.y /= dist;
                    
                    if (dist < 0.1f) dist = 0.1f;
                    const float G = 0.01f;
                    float forceMagnitude = pi.mass * pj.mass * (G / (dist * dist));
                    
                    if (dist > cullRadius75) {
                        float decay = 1.0f - (dist - cullRadius75) * decaySlope;
                        forceMagnitude *= decay;
                    }
                    
                    force.x += dir.x * forceMagnitude;
                    force.y += dir.y * forceMagnitude;
                }
            }
            
            newParticles[i] = updateParticle(pi, force, params.deltaTime);
        }
    }
};

Simulator* createSimulator() {
    return new MySimulator();
}