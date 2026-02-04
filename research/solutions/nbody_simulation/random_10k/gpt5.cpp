#include "world.h"
#include <omp.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

class MySimulator : public Simulator {
private:
    int numThreads = 0;
    int numParticles = 0;

    // Grid parameters
    float cellSize = 1.0f;
    float minX = 0.0f, minY = 0.0f;
    int gridW = 1, gridH = 1;

    // Data structures for grid
    std::vector<int> particleCellId;   // cell id for each particle
    std::vector<int> particleCellX;    // cell x for each particle
    std::vector<int> particleCellY;    // cell y for each particle
    std::vector<int> cellCounts;       // count of particles per cell
    std::vector<int> cellOffsets;      // start offset of each cell in cellParticles (size gridSize+1)
    std::vector<int> cellParticles;    // particle indices sorted by cell

    inline void buildGrid(const std::vector<Particle> &particles, float cullRadius) {
        numParticles = (int)particles.size();
        if ((int)particleCellId.size() != numParticles) {
            particleCellId.resize(numParticles);
            particleCellX.resize(numParticles);
            particleCellY.resize(numParticles);
        }

        // Compute bounding box
        float minx = std::numeric_limits<float>::infinity();
        float miny = std::numeric_limits<float>::infinity();
        float maxx = -std::numeric_limits<float>::infinity();
        float maxy = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < numParticles; ++i) {
            const auto &p = particles[i].position;
            if (p.x < minx) minx = p.x;
            if (p.y < miny) miny = p.y;
            if (p.x > maxx) maxx = p.x;
            if (p.y > maxy) maxy = p.y;
        }

        minX = minx;
        minY = miny;

        // Choose cell size: half the cull radius to reduce candidate checks
        cellSize = std::max(1e-5f, cullRadius * 0.5f);

        float rangeX = maxx - minx;
        float rangeY = maxy - miny;

        gridW = std::max(1, (int)std::floor(rangeX / cellSize) + 1);
        gridH = std::max(1, (int)std::floor(rangeY / cellSize) + 1);
        const int gridSize = gridW * gridH;

        if ((int)cellCounts.size() != gridSize) {
            cellCounts.assign(gridSize, 0);
            cellOffsets.resize(gridSize + 1);
        } else {
            std::fill(cellCounts.begin(), cellCounts.end(), 0);
        }

        // First pass: count particles per cell
        for (int i = 0; i < numParticles; ++i) {
            float fx = (particles[i].position.x - minX) / cellSize;
            float fy = (particles[i].position.y - minY) / cellSize;
            int ix = (int)std::floor(fx);
            int iy = (int)std::floor(fy);
            if (ix < 0) ix = 0; else if (ix >= gridW) ix = gridW - 1;
            if (iy < 0) iy = 0; else if (iy >= gridH) iy = gridH - 1;
            int cid = ix + iy * gridW;
            particleCellId[i] = cid;
            particleCellX[i] = ix;
            particleCellY[i] = iy;
            cellCounts[cid]++;
        }

        // Prefix sum to get offsets
        cellOffsets[0] = 0;
        for (int c = 0; c < gridSize; ++c) {
            cellOffsets[c + 1] = cellOffsets[c] + cellCounts[c];
        }

        // Prepare vector for particles sorted by cell
        if ((int)cellParticles.size() != numParticles) {
            cellParticles.resize(numParticles);
        }

        // Make a copy of offsets to use as current write positions
        std::vector<int> writePos = cellOffsets;

        // Second pass: fill in cellParticles
        for (int i = 0; i < numParticles; ++i) {
            int cid = particleCellId[i];
            int pos = writePos[cid]++;
            cellParticles[pos] = i;
        }
    }

public:
    void init(int nParticles, StepParameters params) override {
        numThreads = omp_get_max_threads();
        omp_set_num_threads(numThreads);
        numParticles = nParticles;

        particleCellId.clear();
        particleCellX.clear();
        particleCellY.clear();
        cellCounts.clear();
        cellOffsets.clear();
        cellParticles.clear();
    }

    void simulateStep(std::vector<Particle> &particles,
                      std::vector<Particle> &newParticles,
                      StepParameters params) override {
        if ((int)newParticles.size() != (int)particles.size())
            newParticles.resize(particles.size());

        // Build grid for neighbor search
        buildGrid(particles, params.cullRadius);

        const float cullR2 = params.cullRadius * params.cullRadius;
        const int k = (int)std::ceil(params.cullRadius / cellSize); // neighbor range in cells

        // Parallel force computation and integration
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < (int)particles.size(); ++i) {
            const Particle pi = particles[i];
            Vec2 force(0.0f, 0.0f);

            const int ix = particleCellX[i];
            const int iy = particleCellY[i];

            int x0 = ix - k; if (x0 < 0) x0 = 0;
            int x1 = ix + k; if (x1 >= gridW) x1 = gridW - 1;
            int y0 = iy - k; if (y0 < 0) y0 = 0;
            int y1 = iy + k; if (y1 >= gridH) y1 = gridH - 1;

            const float pix = pi.position.x;
            const float piy = pi.position.y;

            for (int yy = y0; yy <= y1; ++yy) {
                int base = yy * gridW;
                for (int xx = x0; xx <= x1; ++xx) {
                    int cid = base + xx;
                    int start = cellOffsets[cid];
                    int end = cellOffsets[cid + 1];
                    for (int idx = start; idx < end; ++idx) {
                        int j = cellParticles[idx];
                        if (j == i) continue;

                        const float dx = particles[j].position.x - pix;
                        const float dy = particles[j].position.y - piy;
                        const float dist2 = dx * dx + dy * dy;
                        if (dist2 < cullR2) {
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