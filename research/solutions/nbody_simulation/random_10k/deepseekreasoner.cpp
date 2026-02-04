#include "world.h"
#include <omp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>

struct QuadNode {
    float x, y, size;
    int count;
    float mass;
    Vec2 com;
    int children[4];
    int firstParticle;
    
    QuadNode(float x, float y, float size) 
        : x(x), y(y), size(size), count(0), mass(0), com{0,0}, 
          firstParticle(-1) {
        children[0] = children[1] = children[2] = children[3] = -1;
    }
};

class MySimulator : public Simulator {
private:
    int numThreads;
    std::vector<QuadNode> quadtree;
    std::vector<int> particleIndices;
    std::vector<int> nodeStack;
    std::vector<int> particleToNode;
    
    static constexpr int MAX_PARTICLES_PER_NODE = 16;
    static constexpr float THETA = 0.5f;
    static constexpr int MAX_TREE_SIZE = 20000;
    
    void initQuadtree() {
        quadtree.clear();
        quadtree.reserve(MAX_TREE_SIZE);
        quadtree.emplace_back(0.0f, 0.0f, 100.0f);
        particleToNode.resize(particleIndices.size(), 0);
    }
    
    int getQuadrant(float px, float py, float cx, float cy) {
        int quad = 0;
        if (px >= cx) quad |= 1;
        if (py >= cy) quad |= 2;
        return quad;
    }
    
    void insertParticle(int pidx, int nodeIdx, const std::vector<Particle>& particles) {
        QuadNode& node = quadtree[nodeIdx];
        const Particle& p = particles[pidx];
        
        if (node.children[0] == -1) {
            if (node.count < MAX_PARTICLES_PER_NODE || node.size <= 1.0f) {
                particleIndices[node.firstParticle + node.count] = pidx;
                node.count++;
                particleToNode[pidx] = nodeIdx;
                return;
            }
            
            float half = node.size * 0.5f;
            float quarter = half * 0.5f;
            quadtree.emplace_back(node.x - quarter, node.y - quarter, half);
            quadtree.emplace_back(node.x + quarter, node.y - quarter, half);
            quadtree.emplace_back(node.x - quarter, node.y + quarter, half);
            quadtree.emplace_back(node.x + quarter, node.y + quarter, half);
            
            for (int i = 0; i < 4; i++) {
                node.children[i] = quadtree.size() - 4 + i;
            }
            
            int oldCount = node.count;
            int oldFirst = node.firstParticle;
            node.count = 0;
            
            for (int i = 0; i < oldCount; i++) {
                int oldPidx = particleIndices[oldFirst + i];
                insertParticle(oldPidx, nodeIdx, particles);
            }
            
            insertParticle(pidx, nodeIdx, particles);
        } else {
            int quad = getQuadrant(p.position.x, p.position.y, node.x, node.y);
            insertParticle(pidx, node.children[quad], particles);
        }
    }
    
    void computeNodeProperties(int nodeIdx, const std::vector<Particle>& particles) {
        QuadNode& node = quadtree[nodeIdx];
        
        if (node.children[0] != -1) {
            node.mass = 0;
            node.com = {0, 0};
            node.count = 0;
            
            for (int i = 0; i < 4; i++) {
                computeNodeProperties(node.children[i], particles);
                QuadNode& child = quadtree[node.children[i]];
                if (child.mass > 0) {
                    node.mass += child.mass;
                    node.com.x += child.com.x * child.mass;
                    node.com.y += child.com.y * child.mass;
                    node.count += child.count;
                }
            }
            
            if (node.mass > 0) {
                node.com.x /= node.mass;
                node.com.y /= node.mass;
            }
        } else {
            node.mass = 0;
            node.com = {0, 0};
            
            for (int i = 0; i < node.count; i++) {
                int pidx = particleIndices[node.firstParticle + i];
                const Particle& p = particles[pidx];
                node.mass += p.mass;
                node.com.x += p.position.x * p.mass;
                node.com.y += p.position.y * p.mass;
            }
            
            if (node.mass > 0) {
                node.com.x /= node.mass;
                node.com.y /= node.mass;
            }
        }
    }
    
    void collectNearbyParticles(int nodeIdx, float px, float py, float radius, 
                                std::vector<int>& result, const std::vector<Particle>& particles) {
        const QuadNode& node = quadtree[nodeIdx];
        
        float dx = node.com.x - px;
        float dy = node.com.y - py;
        float dist = std::sqrt(dx*dx + dy*dy);
        
        if (node.children[0] == -1 || (node.size / dist < THETA)) {
            if (node.children[0] == -1) {
                for (int i = 0; i < node.count; i++) {
                    result.push_back(particleIndices[node.firstParticle + i]);
                }
            } else {
                Particle fake;
                fake.position = node.com;
                fake.mass = node.mass;
                result.push_back(-nodeIdx - 1);
            }
        } else {
            for (int i = 0; i < 4; i++) {
                if (node.children[i] != -1) {
                    collectNearbyParticles(node.children[i], px, py, radius, result, particles);
                }
            }
        }
    }

public:
    void init(int numParticles, StepParameters params) override {
        numThreads = omp_get_max_threads();
        omp_set_num_threads(numThreads);
        
        particleIndices.resize(numParticles);
        for (int i = 0; i < numParticles; i++) {
            particleIndices[i] = i;
        }
        
        nodeStack.reserve(256);
    }

    void simulateStep(std::vector<Particle> &particles,
                      std::vector<Particle> &newParticles,
                      StepParameters params) override {
        if (particles.empty()) return;
        
        initQuadtree();
        
        int requiredSize = particles.size() * 2;
        if (particleIndices.size() < requiredSize) {
            particleIndices.resize(requiredSize);
        }
        
        quadtree[0].firstParticle = 0;
        quadtree[0].count = 0;
        
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int i = 0; i < (int)particles.size(); i++) {
                insertParticle(i, 0, particles);
            }
        }
        
        computeNodeProperties(0, particles);
        
        #pragma omp parallel
        {
            std::vector<int> nearby;
            nearby.reserve(512);
            
            #pragma omp for schedule(dynamic, 32)
            for (int i = 0; i < (int)particles.size(); i++) {
                const Particle& pi = particles[i];
                Vec2 force = {0, 0};
                
                nearby.clear();
                collectNearbyParticles(0, pi.position.x, pi.position.y, params.cullRadius, nearby, particles);
                
                for (int idx : nearby) {
                    if (idx >= 0) {
                        if (idx == i) continue;
                        const Particle& pj = particles[idx];
                        float dx = pi.position.x - pj.position.x;
                        float dy = pi.position.y - pj.position.y;
                        float dist2 = dx*dx + dy*dy;
                        if (dist2 < params.cullRadius * params.cullRadius) {
                            force += computeForce(pi, pj, params.cullRadius);
                        }
                    } else {
                        int nodeIdx = -idx - 1;
                        const QuadNode& node = quadtree[nodeIdx];
                        if (node.mass > 0) {
                            Particle fake;
                            fake.id = -1;
                            fake.mass = node.mass;
                            fake.position = node.com;
                            fake.velocity = {0, 0};
                            
                            float dx = pi.position.x - fake.position.x;
                            float dy = pi.position.y - fake.position.y;
                            float dist2 = dx*dx + dy*dy;
                            if (dist2 < params.cullRadius * params.cullRadius && dist2 > 1e-6f) {
                                force += computeForce(pi, fake, params.cullRadius);
                            }
                        }
                    }
                }
                
                newParticles[i] = updateParticle(pi, force, params.deltaTime);
            }
        }
    }
};

Simulator* createSimulator() {
    return new MySimulator();
}