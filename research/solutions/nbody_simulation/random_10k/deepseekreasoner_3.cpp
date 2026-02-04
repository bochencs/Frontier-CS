#include "world.h"
#include <omp.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>
#include <immintrin.h>

class MySimulator : public Simulator {
private:
    struct QuadNode {
        float center_x, center_y, half_width;
        int start, end;
        int child[4];
        bool is_leaf;
        float total_mass;
        float mass_x, mass_y;
    };

    struct ParticleData {
        float x, y, mass;
        int id;
        int cell_x, cell_y;
    };

    int numThreads = 16;
    int grid_size;
    float cell_size;
    std::vector<std::vector<int>> grid;
    std::vector<ParticleData> pdata;
    std::vector<int> cell_offsets;
    std::vector<int> cell_counts;
    std::vector<int> particle_indices;
    
    // Quad-tree buffers
    std::vector<QuadNode> nodes;
    std::vector<int> node_particles;
    std::vector<int> node_stack;
    int max_depth = 8;
    int leaf_capacity = 16;
    
    // Pre-allocated force buffers for OpenMP
    std::vector<std::vector<float>> force_buffers_x;
    std::vector<std::vector<float>> force_buffers_y;

public:
    void init(int numParticles, StepParameters params) override {
        omp_set_num_threads(numThreads);
        
        // Initialize spatial grid
        grid_size = static_cast<int>(100.0f / params.cullRadius) + 1;
        cell_size = 100.0f / grid_size;
        
        grid.resize(grid_size * grid_size);
        pdata.resize(numParticles);
        cell_offsets.resize(grid_size * grid_size + 1);
        cell_counts.resize(grid_size * grid_size, 0);
        particle_indices.resize(numParticles);
        
        // Initialize quad-tree buffers
        nodes.reserve(numParticles * 2 / leaf_capacity);
        node_particles.reserve(numParticles);
        
        // Initialize force buffers for each thread
        force_buffers_x.resize(numThreads);
        force_buffers_y.resize(numThreads);
        for (int t = 0; t < numThreads; ++t) {
            force_buffers_x[t].resize(numParticles, 0.0f);
            force_buffers_y[t].resize(numParticles, 0.0f);
        }
        
        node_stack.reserve(64);
    }

    void simulateStep(std::vector<Particle> &particles,
                      std::vector<Particle> &newParticles,
                      StepParameters params) override {
        const int n = particles.size();
        const float cull2 = params.cullRadius * params.cullRadius;
        const float cull75 = params.cullRadius * 0.75f;
        const float inv_cell_size = 1.0f / cell_size;
        
        // Phase 1: Build spatial grid
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            const auto &p = particles[i];
            int cx = static_cast<int>(p.position.x * inv_cell_size);
            int cy = static_cast<int>(p.position.y * inv_cell_size);
            cx = std::clamp(cx, 0, grid_size - 1);
            cy = std::clamp(cy, 0, grid_size - 1);
            pdata[i] = {p.position.x, p.position.y, p.mass, i, cx, cy};
        }
        
        // Count particles per cell
        std::fill(cell_counts.begin(), cell_counts.end(), 0);
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            int cell_idx = pdata[i].cell_y * grid_size + pdata[i].cell_x;
            #pragma omp atomic
            cell_counts[cell_idx]++;
        }
        
        // Build prefix sum
        cell_offsets[0] = 0;
        for (int i = 0; i < grid_size * grid_size; ++i) {
            cell_offsets[i + 1] = cell_offsets[i] + cell_counts[i];
        }
        
        // Fill grid with particle indices
        std::vector<int> temp_counts = cell_counts;
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            int cell_idx = pdata[i].cell_y * grid_size + pdata[i].cell_x;
            int pos = cell_offsets[cell_idx] + (--temp_counts[cell_idx]);
            particle_indices[pos] = i;
        }
        
        // Build quad-tree
        buildQuadTree(particles, params.cullRadius);
        
        // Phase 2: Compute forces using quad-tree with SIMD optimization
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto &fx = force_buffers_x[tid];
            auto &fy = force_buffers_y[tid];
            std::fill(fx.begin(), fx.end(), 0.0f);
            std::fill(fy.begin(), fy.end(), 0.0f);
            
            #pragma omp for schedule(dynamic, 32)
            for (int i = 0; i < n; ++i) {
                const auto &pi = particles[i];
                float fx_sum = 0.0f, fy_sum = 0.0f;
                
                // Use quad-tree for distant particles
                node_stack.clear();
                node_stack.push_back(0);
                
                while (!node_stack.empty()) {
                    int node_idx = node_stack.back();
                    node_stack.pop_back();
                    const QuadNode &node = nodes[node_idx];
                    
                    if (node.is_leaf) {
                        // Process leaf node particles
                        for (int j = node.start; j < node.end; ++j) {
                            int pj_idx = node_particles[j];
                            if (pj_idx == i) continue;
                            const auto &pj = particles[pj_idx];
                            
                            float dx = pj.position.x - pi.position.x;
                            float dy = pj.position.y - pi.position.y;
                            float dist2 = dx * dx + dy * dy;
                            
                            if (dist2 < 1e-6f || dist2 > cull2) continue;
                            
                            float dist = std::sqrt(dist2);
                            if (dist < 0.1f) dist = 0.1f;
                            
                            float inv_dist = 1.0f / dist;
                            dx *= inv_dist;
                            dy *= inv_dist;
                            
                            float G = 0.01f;
                            float force_mag = pi.mass * pj.mass * G / dist2;
                            
                            if (dist > cull75) {
                                float decay = 1.0f - (dist - cull75) / (params.cullRadius * 0.25f);
                                force_mag *= decay;
                            }
                            
                            fx_sum += dx * force_mag;
                            fy_sum += dy * force_mag;
                        }
                    } else {
                        // Check if we can approximate with node center of mass
                        float dx = node.mass_x / node.total_mass - pi.position.x;
                        float dy = node.mass_y / node.total_mass - pi.position.y;
                        float dist2 = dx * dx + dy * dy;
                        float node_width = node.half_width * 2.0f;
                        
                        // Barnes-Hut criterion: node_width/dist < theta
                        if (node_width * node_width < dist2 * 0.5f && dist2 < cull2) {
                            // Use approximation
                            float dist = std::sqrt(dist2);
                            if (dist < 0.1f) dist = 0.1f;
                            
                            float inv_dist = 1.0f / dist;
                            dx *= inv_dist;
                            dy *= inv_dist;
                            
                            float G = 0.01f;
                            float force_mag = pi.mass * node.total_mass * G / dist2;
                            
                            if (dist > cull75) {
                                float decay = 1.0f - (dist - cull75) / (params.cullRadius * 0.25f);
                                force_mag *= decay;
                            }
                            
                            fx_sum += dx * force_mag;
                            fy_sum += dy * force_mag;
                        } else {
                            // Need to traverse children
                            for (int c = 0; c < 4; ++c) {
                                if (node.child[c] != -1) {
                                    node_stack.push_back(node.child[c]);
                                }
                            }
                        }
                    }
                }
                
                // Add near-field forces from spatial grid (3x3 neighborhood)
                int cx = pdata[i].cell_x;
                int cy = pdata[i].cell_y;
                
                for (int dy = -1; dy <= 1; ++dy) {
                    int ny = cy + dy;
                    if (ny < 0 || ny >= grid_size) continue;
                    
                    for (int dx = -1; dx <= 1; ++dx) {
                        int nx = cx + dx;
                        if (nx < 0 || nx >= grid_size) continue;
                        
                        int cell_idx = ny * grid_size + nx;
                        int start = cell_offsets[cell_idx];
                        int end = cell_offsets[cell_idx + 1];
                        
                        for (int idx = start; idx < end; ++idx) {
                            int j = particle_indices[idx];
                            if (j <= i) continue;
                            
                            const auto &pj = particles[j];
                            float dx_ij = pj.position.x - pi.position.x;
                            float dy_ij = pj.position.y - pi.position.y;
                            float dist2 = dx_ij * dx_ij + dy_ij * dy_ij;
                            
                            if (dist2 < 1e-6f || dist2 > cull2) continue;
                            
                            float dist = std::sqrt(dist2);
                            if (dist < 0.1f) dist = 0.1f;
                            
                            float inv_dist = 1.0f / dist;
                            dx_ij *= inv_dist;
                            dy_ij *= inv_dist;
                            
                            float G = 0.01f;
                            float force_mag = pi.mass * pj.mass * G / dist2;
                            
                            if (dist > cull75) {
                                float decay = 1.0f - (dist - cull75) / (params.cullRadius * 0.25f);
                                force_mag *= decay;
                            }
                            
                            // Apply Newton's 3rd law: add to both particles
                            fx[i] += dx_ij * force_mag;
                            fy[i] += dy_ij * force_mag;
                            fx[j] -= dx_ij * force_mag;
                            fy[j] -= dy_ij * force_mag;
                        }
                    }
                }
                
                fx[i] += fx_sum;
                fy[i] += fy_sum;
            }
        }
        
        // Phase 3: Reduce forces from all threads
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            Vec2 total_force(0.0f, 0.0f);
            for (int t = 0; t < numThreads; ++t) {
                total_force.x += force_buffers_x[t][i];
                total_force.y += force_buffers_y[t][i];
            }
            newParticles[i] = updateParticle(particles[i], total_force, params.deltaTime);
        }
    }

private:
    void buildQuadTree(const std::vector<Particle> &particles, float cull_radius) {
        nodes.clear();
        node_particles.clear();
        
        // Create root node covering entire space
        QuadNode root;
        root.center_x = 50.0f;
        root.center_y = 50.0f;
        root.half_width = 50.0f;
        root.start = 0;
        root.end = particles.size();
        root.is_leaf = true;
        root.total_mass = 0.0f;
        root.mass_x = 0.0f;
        root.mass_y = 0.0f;
        root.child[0] = root.child[1] = root.child[2] = root.child[3] = -1;
        
        nodes.push_back(root);
        
        // Add all particles to root initially
        node_particles.resize(particles.size());
        for (size_t i = 0; i < particles.size(); ++i) {
            node_particles[i] = i;
            
            // Update root's center of mass
            nodes[0].total_mass += particles[i].mass;
            nodes[0].mass_x += particles[i].mass * particles[i].position.x;
            nodes[0].mass_y += particles[i].mass * particles[i].position.y;
        }
        
        // Recursively split nodes
        std::vector<int> node_queue;
        node_queue.push_back(0);
        
        while (!node_queue.empty()) {
            int node_idx = node_queue.back();
            node_queue.pop_back();
            QuadNode &node = nodes[node_idx];
            
            if (node.end - node.start <= leaf_capacity || 
                node.half_width < cull_radius * 0.1f) {
                continue;  // Keep as leaf
            }
            
            // Split the node
            float hw = node.half_width * 0.5f;
            float cx = node.center_x;
            float cy = node.center_y;
            
            // Create child nodes
            int child_indices[4];
            for (int i = 0; i < 4; ++i) {
                QuadNode child;
                child.center_x = cx + ((i & 1) ? hw : -hw);
                child.center_y = cy + ((i & 2) ? hw : -hw);
                child.half_width = hw;
                child.start = node.start;
                child.end = node.start;
                child.is_leaf = true;
                child.total_mass = 0.0f;
                child.mass_x = 0.0f;
                child.mass_y = 0.0f;
                child.child[0] = child.child[1] = child.child[2] = child.child[3] = -1;
                
                nodes.push_back(child);
                child_indices[i] = nodes.size() - 1;
                node.child[i] = child_indices[i];
            }
            
            // Partition particles into children
            std::vector<int> counts(4, 0);
            for (int i = node.start; i < node.end; ++i) {
                int p_idx = node_particles[i];
                const auto &p = particles[p_idx];
                
                int quadrant = 0;
                if (p.position.x > cx) quadrant |= 1;
                if (p.position.y > cy) quadrant |= 2;
                
                counts[quadrant]++;
            }
            
            // Set child start positions
            int offset = node.start;
            for (int i = 0; i < 4; ++i) {
                nodes[child_indices[i]].start = offset;
                offset += counts[i];
                nodes[child_indices[i]].end = offset;
            }
            
            // Reset counts for placement
            std::vector<int> pos(4);
            for (int i = 0; i < 4; ++i) {
                pos[i] = nodes[child_indices[i]].start;
            }
            
            // Place particles in children and compute centers of mass
            for (int i = node.start; i < node.end; ++i) {
                int p_idx = node_particles[i];
                const auto &p = particles[p_idx];
                
                int quadrant = 0;
                if (p.position.x > cx) quadrant |= 1;
                if (p.position.y > cy) quadrant |= 2;
                
                // Place particle in child's range
                node_particles[pos[quadrant]] = p_idx;
                pos[quadrant]++;
                
                // Update child's center of mass
                QuadNode &child = nodes[child_indices[quadrant]];
                child.total_mass += p.mass;
                child.mass_x += p.mass * p.position.x;
                child.mass_y += p.mass * p.position.y;
            }
            
            // Mark node as non-leaf
            node.is_leaf = false;
            
            // Add children to queue for potential further splitting
            for (int i = 0; i < 4; ++i) {
                if (nodes[child_indices[i]].end - nodes[child_indices[i]].start > 0) {
                    node_queue.push_back(child_indices[i]);
                }
            }
        }
    }
};

Simulator* createSimulator() {
    return new MySimulator();
}