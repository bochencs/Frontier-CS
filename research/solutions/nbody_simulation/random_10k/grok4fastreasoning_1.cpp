#include "world.h"
#include <omp.h>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>

struct QuadTree {
    float minX, minY, maxX, maxY;
    std::vector<int> indices;
    std::unique_ptr<QuadTree> nw, ne, sw, se;

    QuadTree(float x0, float y0, float x1, float y1)
        : minX(x0), minY(y0), maxX(x1), maxY(y1) {}

    bool isLeaf() const { return !nw; }

    void insert(int idx, const std::vector<Particle>& parts) {
        if (isLeaf()) {
            bool degenerate = (maxX - minX < 1e-4f) || (maxY - minY < 1e-4f);
            if (indices.size() < 4 || degenerate) {
                indices.push_back(idx);
                return;
            }
            float midX = 0.5f * (minX + maxX);
            float midY = 0.5f * (minY + maxY);
            nw = std::make_unique<QuadTree>(minX, midY, midX, maxY);
            ne = std::make_unique<QuadTree>(midX, midY, maxX, maxY);
            sw = std::make_unique<QuadTree>(minX, minY, midX, midY);
            se = std::make_unique<QuadTree>(midX, minY, maxX, midY);
            std::vector<int> old = std::move(indices);
            for (int pidx : old) {
                insertChild(pidx, parts, midX, midY);
            }
            insertChild(idx, parts, midX, midY);
        } else {
            float midX = 0.5f * (minX + maxX);
            float midY = 0.5f * (minY + maxY);
            insertChild(idx, parts, midX, midY);
        }
    }

private:
    void insertChild(int idx, const std::vector<Particle>& parts, float midX, float midY) {
        Vec2 pos = parts[idx].position;
        if (pos.x < midX) {
            if (pos.y < midY) {
                sw->insert(idx, parts);
            } else {
                nw->insert(idx, parts);
            }
        } else {
            if (pos.y < midY) {
                se->insert(idx, parts);
            } else {
                ne->insert(idx, parts);
            }
        }
    }

public:
    void query(const Vec2& center, float radius, std::vector<int>& result, const std::vector<Particle>& parts) const {
        float r2 = radius * radius;
        auto [minD2, maxD2] = dist2ToBox(center);
        if (maxD2 <= r2) {
            collectAll(result);
        } else if (minD2 > r2) {
            return;
        } else {
            if (isLeaf()) {
                for (int idx : indices) {
                    float d2 = (parts[idx].position - center).length2();
                    if (d2 <= r2 && d2 > 1e-6f) {
                        result.push_back(idx);
                    }
                }
            } else {
                if (nw) nw->query(center, radius, result, parts);
                if (ne) ne->query(center, radius, result, parts);
                if (sw) sw->query(center, radius, result, parts);
                if (se) se->query(center, radius, result, parts);
            }
        }
    }

private:
    void collectAll(std::vector<int>& result) const {
        if (isLeaf()) {
            result.insert(result.end(), indices.begin(), indices.end());
        } else {
            if (nw) nw->collectAll(result);
            if (ne) ne->collectAll(result);
            if (sw) sw->collectAll(result);
            if (se) se->collectAll(result);
        }
    }

    std::pair<float, float> dist2ToBox(const Vec2& center) const {
        float cx = std::max(minX, std::min(center.x, maxX));
        float cy = std::max(minY, std::min(center.y, maxY));
        float dx = cx - center.x;
        float dy = cy - center.y;
        float minD2 = dx * dx + dy * dy;

        float maxD2 = 0.0f;
        auto calc_d2 = [&](float px, float py) -> float {
            dx = px - center.x;
            dy = py - center.y;
            return dx * dx + dy * dy;
        };
        maxD2 = calc_d2(minX, minY);
        maxD2 = std::max(maxD2, calc_d2(minX, maxY));
        maxD2 = std::max(maxD2, calc_d2(maxX, minY));
        maxD2 = std::max(maxD2, calc_d2(maxX, maxY));

        return {minD2, maxD2};
    }
};

class MySimulator : public Simulator {
public:
    void init(int numParticles, StepParameters params) override {
        omp_set_num_threads(16);
    }

    void simulateStep(std::vector<Particle> &particles,
                      std::vector<Particle> &newParticles,
                      StepParameters params) override {
        size_t N = particles.size();
        if (N == 0) {
            newParticles.clear();
            return;
        }
        float minX = particles[0].position.x;
        float minY = particles[0].position.y;
        float maxX = minX;
        float maxY = minY;
        for (size_t i = 1; i < N; ++i) {
            float px = particles[i].position.x;
            float py = particles[i].position.y;
            minX = std::min(minX, px);
            minY = std::min(minY, py);
            maxX = std::max(maxX, px);
            maxY = std::max(maxY, py);
        }
        float padding = params.cullRadius * 1.2f;
        minX -= padding;
        minY -= padding;
        maxX += padding;
        maxY += padding;
        QuadTree qt(minX, minY, maxX, maxY);
        for (size_t i = 0; i < N; ++i) {
            qt.insert(static_cast<int>(i), particles);
        }
#pragma omp parallel for schedule(dynamic, 50)
        for (int i = 0; i < static_cast<int>(N); ++i) {
            const Particle& pi = particles[i];
            Vec2 force = {0.0f, 0.0f};
            std::vector<int> neighbors;
            qt.query(pi.position, params.cullRadius, neighbors, particles);
            for (int j : neighbors) {
                force += computeForce(pi, particles[j], params.cullRadius);
            }
            newParticles[i] = updateParticle(pi, force, params.deltaTime);
        }
    }
};

Simulator* createSimulator() {
    return new MySimulator();
}