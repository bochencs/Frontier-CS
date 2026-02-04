#include "world.h"
#include <vector>
#include <cmath>
#include <omp.h>
#include <array>

class MySimulator : public Simulator {
private:
    struct QuadTree {
        float minX, minY, maxX, maxY;
        int capacity;
        bool isDivided;
        std::vector<Particle> particles;
        QuadTree* children[4];

        QuadTree(float mx, float my, float Mx, float My, int cap = 4)
            : minX(mx), minY(my), maxX(Mx), maxY(My), capacity(cap), isDivided(false) {
            std::fill(children, children + 4, nullptr);
        }

        ~QuadTree() {
            for (int i = 0; i < 4; ++i) {
                delete children[i];
            }
        }

        bool inBoundary(const Particle& p) const {
            return p.position.x >= minX - 1e-5f && p.position.x <= maxX + 1e-5f &&
                   p.position.y >= minY - 1e-5f && p.position.y <= maxY + 1e-5f;
        }

        int getIndex(const Particle& p) const {
            float midX = (minX + maxX) / 2.0f;
            float midY = (minY + maxY) / 2.0f;
            int idx = 0;
            if (p.position.x >= midX) idx += 1;
            if (p.position.y >= midY) idx += 2;
            return idx;
        }

        void subdivide() {
            float midX = (minX + maxX) / 2.0f;
            float midY = (minY + maxY) / 2.0f;
            children[0] = new QuadTree(minX, minY, midX, midY, capacity); // SW
            children[1] = new QuadTree(midX, minY, maxX, midY, capacity); // SE
            children[2] = new QuadTree(minX, midY, midX, maxY, capacity); // NW
            children[3] = new QuadTree(midX, midY, maxX, maxY, capacity); // NE
            isDivided = true;
            for (const auto& existing : particles) {
                int idx = getIndex(existing);
                children[idx]->insert(existing);
            }
            particles.clear();
        }

        void insert(const Particle& p) {
            if (!inBoundary(p)) return;
            if (!isDivided && particles.size() < static_cast<size_t>(capacity)) {
                particles.push_back(p);
                return;
            }
            if (!isDivided) {
                subdivide();
            }
            int idx = getIndex(p);
            children[idx]->insert(p);
        }

        float distToClosest(const Vec2& point) const {
            float dx = 0.0f;
            if (point.x < minX) dx = minX - point.x;
            else if (point.x > maxX) dx = point.x - maxX;
            float dy = 0.0f;
            if (point.y < minY) dy = minY - point.y;
            else if (point.y > maxY) dy = point.y - maxY;
            return std::sqrt(dx * dx + dy * dy);
        }

        float maxDistTo(const Vec2& point) const {
            float d1 = (Vec2{minX, minY} - point).length();
            float d2 = (Vec2{minX, maxY} - point).length();
            float d3 = (Vec2{maxX, minY} - point).length();
            float d4 = (Vec2{maxX, maxY} - point).length();
            return std::max({d1, d2, d3, d4});
        }

        void getAllParticles(std::vector<Particle>& res) const {
            if (isDivided) {
                for (int i = 0; i < 4; ++i) {
                    if (children[i]) {
                        children[i]->getAllParticles(res);
                    }
                }
            } else {
                for (const auto& p : particles) {
                    res.push_back(p);
                }
            }
        }

        void query(const Vec2& center, float radius, std::vector<Particle>& result) const {
            float closestD = distToClosest(center);
            if (closestD > radius) return;
            float farthestD = maxDistTo(center);
            if (farthestD <= radius + 1e-5f) {
                getAllParticles(result);
                return;
            }
            if (!isDivided) {
                for (const auto& p : particles) {
                    if ((p.position - center).length() <= radius + 1e-5f) {
                        result.push_back(p);
                    }
                }
            } else {
                for (int i = 0; i < 4; ++i) {
                    if (children[i]) {
                        children[i]->query(center, radius, result);
                    }
                }
            }
        }
    };

    int numThreads = 16;

public:
    void init(int numParticles, StepParameters params) override {
        omp_set_num_threads(numThreads);
    }

    void simulateStep(std::vector<Particle> &particles,
                      std::vector<Particle> &newParticles,
                      StepParameters params) override {
        QuadTree* qt = new QuadTree(0.0f, 0.0f, 100.0f, 100.0f, 4);
        for (const auto& p : particles) {
            qt->insert(p);
        }
        #pragma omp parallel for schedule(dynamic, 256)
        for (size_t i = 0; i < particles.size(); ++i) {
            const Particle& pi = particles[i];
            std::vector<Particle> nearby;
            nearby.reserve(2000);
            qt->query(pi.position, params.cullRadius, nearby);
            Vec2 force{0.0f, 0.0f};
            for (const auto& pj : nearby) {
                force += computeForce(pi, pj, params.cullRadius);
            }
            newParticles[i] = updateParticle(pi, force, params.deltaTime);
        }
        delete qt;
    }
};

Simulator* createSimulator() {
    return new MySimulator();
}