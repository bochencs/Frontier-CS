#include "world.h"
#include <omp.h>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>

struct QuadTree {
    float minX, minY, maxX, maxY;
    bool isLeaf;
    std::vector<Particle> particles;
    std::unique_ptr<QuadTree> children[4];

    QuadTree(float mx, float my, float Mx, float My)
        : minX(mx), minY(my), maxX(Mx), maxY(My), isLeaf(true) {}

    int getQuadrant(const Vec2& pos) const {
        float midX = (minX + maxX) * 0.5f;
        float midY = (minY + maxY) * 0.5f;
        if (pos.x < midX) {
            return (pos.y < midY) ? 0 : 2;
        } else {
            return (pos.y < midY) ? 1 : 3;
        }
    }

    void subdivide() {
        float midX = (minX + maxX) * 0.5f;
        float midY = (minY + maxY) * 0.5f;
        children[0] = std::make_unique<QuadTree>(minX, minY, midX, midY);
        children[1] = std::make_unique<QuadTree>(midX, minY, maxX, midY);
        children[2] = std::make_unique<QuadTree>(minX, midY, midX, maxY);
        children[3] = std::make_unique<QuadTree>(midX, midY, maxX, maxY);
        isLeaf = false;
    }

    void insert(const Particle& p) {
        if (p.position.x < minX || p.position.x >= maxX ||
            p.position.y < minY || p.position.y >= maxY) {
            return;
        }
        if (!isLeaf) {
            int idx = getQuadrant(p.position);
            children[idx]->insert(p);
            return;
        }
        particles.push_back(p);
        if (particles.size() > 4) {
            subdivide();
            std::vector<Particle> temp = std::move(particles);
            particles.clear();
            for (const auto& pp : temp) {
                int idx = getQuadrant(pp.position);
                children[idx]->insert(pp);
            }
        }
    }

    void query(const Vec2& center, float radius, std::vector<Particle>& result) const {
        float closestX = std::max(minX, std::min(center.x, maxX));
        float closestY = std::max(minY, std::min(center.y, maxY));
        float dx = center.x - closestX;
        float dy = center.y - closestY;
        float dist2 = dx * dx + dy * dy;
        float r2 = radius * radius;
        if (dist2 > r2) return;
        if (isLeaf) {
            for (const auto& p : particles) {
                Vec2 diff = p.position - center;
                if (diff.length2() <= r2) {
                    result.push_back(p);
                }
            }
        } else {
            for (int i = 0; i < 4; ++i) {
                children[i]->query(center, radius, result);
            }
        }
    }
};

class MySimulator : public Simulator {
private:
    int numThreads = 16;

public:
    void init(int numParticles, StepParameters params) override {
        omp_set_num_threads(numThreads);
    }

    void simulateStep(std::vector<Particle>& particles,
                      std::vector<Particle>& newParticles,
                      StepParameters params) override {
        float worldMinX = 1e30f, worldMinY = 1e30f, worldMaxX = -1e30f, worldMaxY = -1e30f;
        for (const auto& p : particles) {
            worldMinX = std::min(worldMinX, p.position.x);
            worldMinY = std::min(worldMinY, p.position.y);
            worldMaxX = std::max(worldMaxX, p.position.x);
            worldMaxY = std::max(worldMaxY, p.position.y);
        }
        float margin = params.cullRadius * 2.0f;
        worldMinX -= margin;
        worldMinY -= margin;
        worldMaxX += margin;
        worldMaxY += margin;

        QuadTree tree(worldMinX, worldMinY, worldMaxX, worldMaxY);
        for (const auto& p : particles) {
            tree.insert(p);
        }

        #pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < (int)particles.size(); ++i) {
            const auto& pi = particles[i];
            std::vector<Particle> neighbors;
            tree.query(pi.position, params.cullRadius, neighbors);
            Vec2 force(0.0f, 0.0f);
            for (const auto& pj : neighbors) {
                if (pj.id != pi.id) {
                    force += computeForce(pi, pj, params.cullRadius);
                }
            }
            newParticles[i] = updateParticle(pi, force, params.deltaTime);
        }
    }
};

Simulator* createSimulator() {
    return new MySimulator();
}