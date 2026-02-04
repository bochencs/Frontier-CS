#include "world.h"
#include <vector>
#include <cmath>
#include <omp.h>
#include <algorithm>
#include <limits>

class MySimulator : public Simulator {
private:
    int numThreads = 16;

public:
    void init(int numParticles, StepParameters params) override {
        omp_set_num_threads(numThreads);
    }

    void simulateStep(std::vector<Particle> &particles,
                      std::vector<Particle> &newParticles,
                      StepParameters params) override {
        size_t N = particles.size();
        if (N == 0) return;

        float min_x = std::numeric_limits<float>::infinity();
        float max_x = -std::numeric_limits<float>::infinity();
        float min_y = min_x;
        float max_y = max_x;

        for (const auto& p : particles) {
            min_x = std::min(min_x, p.position.x);
            max_x = std::max(max_x, p.position.x);
            min_y = std::min(min_y, p.position.y);
            max_y = std::max(max_y, p.position.y);
        }

        float cell_size = params.cullRadius;
        if (cell_size <= 0.0f) cell_size = 1.0f;

        float range_x = max_x - min_x;
        float range_y = max_y - min_y;

        int nx = static_cast<int>(std::ceil(range_x / cell_size)) + 2;
        int ny = static_cast<int>(std::ceil(range_y / cell_size)) + 2;

        size_t total_cells = static_cast<size_t>(nx) * ny;
        std::vector<std::vector<size_t>> grid(total_cells);

        for (size_t i = 0; i < N; ++i) {
            float x = particles[i].position.x;
            float y = particles[i].position.y;
            int ix = static_cast<int>(std::floor((x - min_x) / cell_size));
            int iy = static_cast<int>(std::floor((y - min_y) / cell_size));
            ix = std::max(0, std::min(ix, nx - 1));
            iy = std::max(0, std::min(iy, ny - 1));
            size_t cell = static_cast<size_t>(ix) + static_cast<size_t>(iy) * static_cast<size_t>(nx);
            grid[cell].push_back(i);
        }

        float r2 = params.cullRadius * params.cullRadius;

#pragma omp parallel for schedule(dynamic, 32)
        for (size_t i = 0; i < N; ++i) {
            const Particle& pi = particles[i];
            Vec2 force = {0.0f, 0.0f};

            float x = pi.position.x;
            float y = pi.position.y;
            int ix = static_cast<int>(std::floor((x - min_x) / cell_size));
            int iy = static_cast<int>(std::floor((y - min_y) / cell_size));
            ix = std::max(0, std::min(ix, nx - 1));
            iy = std::max(0, std::min(iy, ny - 1));

            for (int di = -1; di <= 1; ++di) {
                int nix = ix + di;
                if (nix < 0 || nix >= nx) continue;
                for (int dj = -1; dj <= 1; ++dj) {
                    int niy = iy + dj;
                    if (niy < 0 || niy >= ny) continue;
                    size_t ncell = static_cast<size_t>(nix) + static_cast<size_t>(niy) * static_cast<size_t>(nx);
                    for (size_t j : grid[ncell]) {
                        if (j == i) continue;
                        Vec2 diff = pi.position - particles[j].position;
                        if (diff.length2() < r2) {
                            force += computeForce(pi, particles[j], params.cullRadius);
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