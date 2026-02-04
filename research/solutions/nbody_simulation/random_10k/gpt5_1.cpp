#include "world.h"
#include <omp.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <limits>

class MySimulator : public Simulator {
private:
    int numThreads = 16;

    // Grid parameters
    float gridSize = 1.0f;
    int neighborRange = 1;
    float invGridSize = 1.0f;

    // Persistent buffers to minimize reallocations
    std::vector<int> cellCounts;
    std::vector<int> cellStarts;
    std::vector<int> cellEnds;
    std::vector<int> cellParticles;

    // Helper to compute integer cell coordinates safely
    inline int clampInt(int v, int lo, int hi) const {
        return v < lo ? lo : (v > hi ? hi : v);
    }

public:
    void init(int numParticles, StepParameters params) override {
        numThreads = std::min(16, omp_get_max_threads());
        omp_set_num_threads(numThreads);

        // Choose gridSize relative to cull radius to reduce candidate pairs.
        // Using cullRadius/4 yields neighborRange=4 (9x9 cells) on typical inputs.
        float div = 4.0f;
        gridSize = std::max(params.cullRadius / div, 1e-3f);
        invGridSize = 1.0f / gridSize;
        neighborRange = std::max(1, (int)std::ceil(params.cullRadius / gridSize));
        cellParticles.resize(numParticles);
        cellCounts.clear();
        cellStarts.clear();
        cellEnds.clear();
    }

    void simulateStep(std::vector<Particle> &particles,
                      std::vector<Particle> &newParticles,
                      StepParameters params) override {
        const int N = (int)particles.size();
        if ((int)cellParticles.size() < N) cellParticles.resize(N);

        // Compute bounding box
        float minx = std::numeric_limits<float>::infinity();
        float miny = std::numeric_limits<float>::infinity();
        float maxx = -std::numeric_limits<float>::infinity();
        float maxy = -std::numeric_limits<float>::infinity();

        for (int i = 0; i < N; ++i) {
            const float x = particles[i].position.x;
            const float y = particles[i].position.y;
            if (x < minx) minx = x;
            if (y < miny) miny = y;
            if (x > maxx) maxx = x;
            if (y > maxy) maxy = y;
        }

        // Expand slightly to avoid boundary issues
        const float eps = 1e-4f;
        minx -= eps; miny -= eps;
        maxx += eps; maxy += eps;

        // Determine grid dimensions
        int nx = std::max(1, (int)std::floor((maxx - minx) * invGridSize) + 1);
        int ny = std::max(1, (int)std::floor((maxy - miny) * invGridSize) + 1);
        int nCells = nx * ny;

        // Prepare grid structures
        cellCounts.assign(nCells, 0);
        cellStarts.assign(nCells, 0);
        cellEnds.assign(nCells, 0);

        // Count particles per cell
        for (int i = 0; i < N; ++i) {
            int cx = (int)((particles[i].position.x - minx) * invGridSize);
            int cy = (int)((particles[i].position.y - miny) * invGridSize);
            if (cx < 0) cx = 0; else if (cx >= nx) cx = nx - 1;
            if (cy < 0) cy = 0; else if (cy >= ny) cy = ny - 1;
            int cell = cy * nx + cx;
            cellCounts[cell]++;
        }

        // Prefix sum to get starts
        int total = 0;
        for (int c = 0; c < nCells; ++c) {
            cellStarts[c] = total;
            total += cellCounts[c];
        }
        // Ensure container size
        if ((int)cellParticles.size() < total) cellParticles.resize(total);

        // Temp write positions (copy of starts)
        std::vector<int> writePos = cellStarts;

        // Fill cell particles
        for (int i = 0; i < N; ++i) {
            int cx = (int)((particles[i].position.x - minx) * invGridSize);
            int cy = (int)((particles[i].position.y - miny) * invGridSize);
            if (cx < 0) cx = 0; else if (cx >= nx) cx = nx - 1;
            if (cy < 0) cy = 0; else if (cy >= ny) cy = ny - 1;
            int cell = cy * nx + cx;
            int &w = writePos[cell];
            cellParticles[w++] = i;
        }

        // Compute cell ends
        for (int c = 0; c < nCells; ++c) {
            cellEnds[c] = cellStarts[c] + cellCounts[c];
        }

        // Perform force computation using grid
        const float cullRadius = params.cullRadius;
        const float cullR2 = cullRadius * cullRadius;
        const float dt = params.deltaTime;

        #pragma omp parallel for schedule(dynamic, 32)
        for (int i = 0; i < N; ++i) {
            const Particle pi = particles[i];
            const float px = pi.position.x;
            const float py = pi.position.y;

            int pcx = (int)((px - minx) * invGridSize);
            int pcy = (int)((py - miny) * invGridSize);
            if (pcx < 0) pcx = 0; else if (pcx >= nx) pcx = nx - 1;
            if (pcy < 0) pcy = 0; else if (pcy >= ny) pcy = ny - 1;

            Vec2 force(0.0f, 0.0f);

            int ry = neighborRange;
            int rx = neighborRange;

            int y0 = pcy - ry;
            int y1 = pcy + ry;
            if (y0 < 0) y0 = 0;
            if (y1 >= ny) y1 = ny - 1;

            int x0 = pcx - rx;
            int x1 = pcx + rx;
            if (x0 < 0) x0 = 0;
            if (x1 >= nx) x1 = nx - 1;

            for (int cy = y0; cy <= y1; ++cy) {
                int base = cy * nx;
                for (int cx = x0; cx <= x1; ++cx) {
                    int cell = base + cx;
                    int start = cellStarts[cell];
                    int end = cellEnds[cell];
                    for (int idx = start; idx < end; ++idx) {
                        int j = cellParticles[idx];
                        if (j == i) continue;
                        const Particle &pj = particles[j];
                        float dx = pj.position.x - px;
                        float dy = pj.position.y - py;
                        float d2 = dx * dx + dy * dy;
                        if (d2 <= cullR2) {
                            force += computeForce(pi, pj, cullRadius);
                        }
                    }
                }
            }

            newParticles[i] = updateParticle(pi, force, dt);
        }
    }
};

Simulator* createSimulator() {
    return new MySimulator();
}