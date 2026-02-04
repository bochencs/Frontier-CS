#include "world.h"
#include <omp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

class GridSimulator : public Simulator {
private:
    int numThreads = 16;
    
    // Grid data structures: Linked List approach
    // cell_head[cell_index] stores the index of the first particle in that cell
    std::vector<int> cell_head;
    // next_particle[particle_index] stores the index of the next particle in the same cell
    std::vector<int> next_particle;

public:
    void init(int numParticles, StepParameters params) override {
        // Detect hardware concurrency
        numThreads = omp_get_max_threads();
        if (numThreads < 1) numThreads = 1;
        omp_set_num_threads(numThreads);
        
        // Pre-allocate arrays
        next_particle.resize(numParticles);
        // Reserve some space for the grid head array
        // Assuming reasonably dense packing, but will resize dynamically
        cell_head.reserve(1024);
    }

    void simulateStep(std::vector<Particle> &particles,
                      std::vector<Particle> &newParticles,
                      StepParameters params) override {
        
        int n = static_cast<int>(particles.size());
        if (n == 0) return;

        // 1. Compute bounds to determine grid size
        float min_x = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest();
        float min_y = std::numeric_limits<float>::max();
        float max_y = std::numeric_limits<float>::lowest();

        // Parallel reduction for bounding box
        #pragma omp parallel for reduction(min:min_x, min_y) reduction(max:max_x, max_y)
        for (int i = 0; i < n; i++) {
            min_x = std::min(min_x, particles[i].position.x);
            max_x = std::max(max_x, particles[i].position.x);
            min_y = std::min(min_y, particles[i].position.y);
            max_y = std::max(max_y, particles[i].position.y);
        }

        // 2. Setup Grid
        float cell_size = params.cullRadius;
        if (cell_size <= 1e-4f) cell_size = 1.0f; // Prevent division by zero

        // Determine grid dimensions
        int grid_w = static_cast<int>((max_x - min_x) / cell_size) + 1;
        int grid_h = static_cast<int>((max_y - min_y) / cell_size) + 1;
        
        long long total_cells = (long long)grid_w * grid_h;
        int num_cells = static_cast<int>(total_cells);

        // Resize grid headers if needed
        if ((int)cell_head.size() < num_cells) {
            cell_head.resize(num_cells);
        }

        // Reset grid headers
        // For large grids, parallel initialization is faster
        if (num_cells > 20000) {
            #pragma omp parallel for
            for (int i = 0; i < num_cells; i++) {
                cell_head[i] = -1;
            }
        } else {
            std::fill(cell_head.begin(), cell_head.begin() + num_cells, -1);
        }

        // 3. Build Grid (Spatial Hashing / Linked List construction)
        // Serial execution avoids synchronization overhead and is fast for 10k particles
        for (int i = 0; i < n; i++) {
            int cx = static_cast<int>((particles[i].position.x - min_x) / cell_size);
            int cy = static_cast<int>((particles[i].position.y - min_y) / cell_size);
            
            // Clamp to grid bounds to handle precision edge cases
            if (cx < 0) cx = 0; else if (cx >= grid_w) cx = grid_w - 1;
            if (cy < 0) cy = 0; else if (cy >= grid_h) cy = grid_h - 1;
            
            int cell_idx = cx + cy * grid_w;
            
            // Insert into linked list at head of cell
            next_particle[i] = cell_head[cell_idx];
            cell_head[cell_idx] = i;
        }

        // 4. Compute Forces
        float r2 = params.cullRadius * params.cullRadius;
        
        // Dynamic schedule handles varying density well
        #pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < n; i++) {
            const Particle& pi = particles[i];
            Vec2 force = {0.0f, 0.0f};
            
            int cx = static_cast<int>((pi.position.x - min_x) / cell_size);
            int cy = static_cast<int>((pi.position.y - min_y) / cell_size);
            
            if (cx < 0) cx = 0; else if (cx >= grid_w) cx = grid_w - 1;
            if (cy < 0) cy = 0; else if (cy >= grid_h) cy = grid_h - 1;

            // Iterate through 3x3 neighbor cells
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = cx + dx;
                    int ny = cy + dy;
                    
                    if (nx >= 0 && nx < grid_w && ny >= 0 && ny < grid_h) {
                        int cell_idx = nx + ny * grid_w;
                        int j = cell_head[cell_idx];
                        
                        // Traverse linked list of particles in this cell
                        while (j != -1) {
                            if (i != j) {
                                const Particle& pj = particles[j];
                                Vec2 diff = pj.position - pi.position;
                                // Optimization: check squared distance to avoid expensive sqrt/force call
                                if (diff.length2() <= r2) {
                                    force += computeForce(pi, pj, params.cullRadius);
                                }
                            }
                            j = next_particle[j];
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