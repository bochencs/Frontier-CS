#include "world.h"
#include <omp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>

// Structure to hold spatial hash references
struct ParticleRef {
    uint32_t hash;
    int id;
};

class GridSimulator : public Simulator {
private:
    // Thread count for the target hardware
    int numThreads = 16;
    
    // Spatial Hashing Constants
    // Using a sufficiently large power of 2 to minimize collisions while fitting in cache
    static const int HASH_SIZE = 16384; 
    static const int HASH_MASK = HASH_SIZE - 1;
    
    // Data buffers
    std::vector<ParticleRef> refs;
    std::vector<int> cellStart;
    std::vector<int> cellEnd;
    
    // Sorted buffer for improved memory locality
    std::vector<Particle> sortedParticles;

public:
    void init(int numParticles, StepParameters params) override {
        omp_set_num_threads(numThreads);
        
        refs.resize(numParticles);
        sortedParticles.resize(numParticles);
        
        cellStart.resize(HASH_SIZE);
        cellEnd.resize(HASH_SIZE);
    }

    void simulateStep(std::vector<Particle> &particles,
                      std::vector<Particle> &newParticles,
                      StepParameters params) override {
        int N = (int)particles.size();
        
        // Ensure buffers are correctly sized
        if ((int)refs.size() != N) {
            refs.resize(N);
            sortedParticles.resize(N);
        }
        
        // Use a cell size of half the cull radius for finer granularity.
        // This reduces the number of unnecessary particle pairs checked compared to cell size = cullRadius.
        // Trade-off: checks 25 neighbor cells (5x5) instead of 9 (3x3), but reduces area from ~56% to ~39% of world.
        float cellSize = params.cullRadius * 0.5f;
        float invCellSize = 1.0f / cellSize;
        
        // 1. Compute Spatial Hashes (Parallel)
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            // Use floor to handle potential negative coordinates gracefully
            int gx = static_cast<int>(std::floor(particles[i].position.x * invCellSize));
            int gy = static_cast<int>(std::floor(particles[i].position.y * invCellSize));
            
            // Hash function with large primes
            uint32_t h = ((gx * 73856093) ^ (gy * 19349663)) & HASH_MASK;
            
            refs[i] = {h, i};
        }
        
        // 2. Sort References (Serial)
        // std::sort is efficient enough for N=10,000 (~0.5ms)
        std::sort(refs.begin(), refs.end(), [](const ParticleRef& a, const ParticleRef& b) {
            return a.hash < b.hash;
        });
        
        // 3. Build Grid Index and Reorder Particles (Mixed)
        // Reset cell starts
        std::fill(cellStart.begin(), cellStart.end(), -1);
        
        // Reorder particles into a temporary buffer for sequential memory access in inner loop
        #pragma omp parallel 
        {
            #pragma omp for
            for (int i = 0; i < N; i++) {
                sortedParticles[i] = particles[refs[i].id];
            }
        }
        
        // Fill start/end indices (Serial - very fast linear scan)
        for (int i = 0; i < N; i++) {
            uint32_t h = refs[i].hash;
            if (cellStart[h] == -1) cellStart[h] = i;
            cellEnd[h] = i + 1;
        }
        
        float cullRadiusSq = params.cullRadius * params.cullRadius;

        // 4. Force Computation (Parallel)
        #pragma omp parallel for schedule(dynamic, 32)
        for (int i = 0; i < N; i++) {
            const Particle& pi = particles[i];
            Vec2 force = {0.0f, 0.0f};
            
            int gx = static_cast<int>(std::floor(pi.position.x * invCellSize));
            int gy = static_cast<int>(std::floor(pi.position.y * invCellSize));
            
            // Iterate over 5x5 neighbor cells (radius covers 2 cells)
            for (int dy = -2; dy <= 2; dy++) {
                for (int dx = -2; dx <= 2; dx++) {
                    uint32_t h = (((gx + dx) * 73856093) ^ ((gy + dy) * 19349663)) & HASH_MASK;
                    
                    int start = cellStart[h];
                    if (start == -1) continue;
                    int end = cellEnd[h];
                    
                    for (int k = start; k < end; k++) {
                        // Check original ID to avoid self-interaction
                        // Access refs[k] is sequential
                        if (refs[k].id == i) continue;
                        
                        // Access sortedParticles[k] is sequential (cache friendly)
                        const Particle& pj = sortedParticles[k];
                        
                        float dx_pos = pi.position.x - pj.position.x;
                        float dy_pos = pi.position.y - pj.position.y;
                        float distSq = dx_pos * dx_pos + dy_pos * dy_pos;
                        
                        // Early exit to avoid expensive sqrt/calls
                        if (distSq < cullRadiusSq) {
                            force += computeForce(pi, pj, params.cullRadius);
                        }
                    }
                }
            }
            
            newParticles[i] = updateParticle(pi, force, params.deltaTime);
        }
    }
};

Simulator* createSimulator() {
    return new GridSimulator();
}