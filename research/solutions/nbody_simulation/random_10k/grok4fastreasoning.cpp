#include "world.h"
#include <omp.h>
#include <vector>
#include <cmath>
#include <algorithm>

class MySimulator : public Simulator {
private:
    int numThreads = 16;
    float worldSize = 100.0f;

public:
    void init(int numParticles, StepParameters params) override {
        omp_set_num_threads(numThreads);
    }

    void simulateStep(std::vector<Particle> &particles,
                      std::vector<Particle> &newParticles,
                      StepParameters params) override {
        int N = particles.size();
        float R = params.cullRadius;
        float cellSize = R / 5.0f;
        if (cellSize < 1e-3f) cellSize = 1e-3f;
        int numCellsX = static_cast<int>(std::ceil(worldSize / cellSize));
        int numCellsY = numCellsX;
        int totalCells = numCellsX * numCellsY;
        std::vector<std::vector<int>> grid(totalCells);

        // Build grid
        for (int i = 0; i < N; ++i) {
            Vec2 pos = particles[i].position;
            int cx = static_cast<int>(std::floor(pos.x / cellSize));
            int cy = static_cast<int>(std::floor(pos.y / cellSize));
            cx = std::max(0, std::min(numCellsX - 1, cx));
            cy = std::max(0, std::min(numCellsY - 1, cy));
            int cellId = cx * numCellsY + cy;
            grid[cellId].push_back(i);
        }

        int k = static_cast<int>(std::ceil(R / cellSize));

#pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < N; ++i) {
            Particle pi = particles[i];
            Vec2 force(0.0f, 0.0f);
            Vec2 pos = pi.position;
            int cx = static_cast<int>(std::floor(pos.x / cellSize));
            int cy = static_cast<int>(std::floor(pos.y / cellSize));
            cx = std::max(0, std::min(numCellsX - 1, cx));
            cy = std::max(0, std::min(numCellsY - 1, cy));
            int minCX = std::max(0, cx - k);
            int maxCX = std::min(numCellsX - 1, cx + k);
            int minCY = std::max(0, cy - k);
            int maxCY = std::min(numCellsY - 1, cy + k);

            for (int dcx = minCX; dcx <= maxCX; ++dcx) {
                for (int dcy = minCY; dcy <= maxCY; ++dcy) {
                    int cellId = dcx * numCellsY + dcy;
                    for (int j : grid[cellId]) {
                        if (j == i) continue;
                        Vec2 diff = particles[j].position - pos;
                        float dist2 = diff.length2();
                        if (dist2 > R * R || dist2 < 1e-6f) continue;
                        force += computeForce(pi, particles[j], R);
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