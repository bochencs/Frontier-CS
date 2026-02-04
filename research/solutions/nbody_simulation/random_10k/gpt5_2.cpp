#include "world.h"
#include <omp.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

class MySimulator : public Simulator {
private:
    int numThreads = 0;
    int nParticles = 0;

    // Grid data
    int gridW = 1, gridH = 1;
    float cellSize = 1.0f;
    float minX = 0.0f, minY = 0.0f;

    // Persistent buffers to reduce allocations
    std::vector<int> cellCounts;           // count per cell
    std::vector<int> cellStarts;           // start index per cell (exclusive prefix sum)
    std::vector<int> cellWritePos;         // temp write positions during fill
    std::vector<int> particleCell;         // cell index per particle
    std::vector<int> sortedParticles;      // particle indices sorted by cell

public:
    void init(int numParticles, StepParameters params) override {
        nParticles = numParticles;
        numThreads = omp_get_max_threads();
        if (numThreads <= 0) numThreads = 1;
        omp_set_num_threads(numThreads);

        // Reserve buffers
        particleCell.resize(nParticles);
        sortedParticles.resize(nParticles);
        // Grid-dependent buffers will be sized in simulateStep
        cellCounts.clear();
        cellStarts.clear();
        cellWritePos.clear();
    }

    void simulateStep(std::vector<Particle> &particles,
                      std::vector<Particle> &newParticles,
                      StepParameters params) override {
        const int N = (int)particles.size();
        if (N == 0) return;

        // 1. Compute bounding box (sequential; cheap for 10k)
        float mnx = std::numeric_limits<float>::infinity();
        float mny = std::numeric_limits<float>::infinity();
        float mxx = -std::numeric_limits<float>::infinity();
        float mxy = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < N; ++i) {
            const auto &p = particles[i].position;
            if (p.x < mnx) mnx = p.x;
            if (p.y < mny) mny = p.y;
            if (p.x > mxx) mxx = p.x;
            if (p.y > mxy) mxy = p.y;
        }

        // Add a tiny epsilon to avoid boundary issues
        const float eps = 1e-4f;
        mnx -= eps; mny -= eps;
        mxx += eps; mxy += eps;

        // 2. Build uniform grid
        cellSize = std::max(params.cullRadius, 1e-6f);
        gridW = std::max(1, (int)((mxx - mnx) / cellSize) + 1);
        gridH = std::max(1, (int)((mxy - mny) / cellSize) + 1);
        minX = mnx; minY = mny;

        const int numCells = gridW * gridH;
        cellCounts.assign(numCells, 0);
        cellStarts.assign(numCells + 1, 0);
        particleCell.resize(N);
        sortedParticles.resize(N);

        // First pass: compute cell per particle and accumulate counts
        for (int i = 0; i < N; ++i) {
            const auto &pos = particles[i].position;
            int cx = (int)((pos.x - minX) / cellSize);
            int cy = (int)((pos.y - minY) / cellSize);
            if (cx < 0) cx = 0; else if (cx >= gridW) cx = gridW - 1;
            if (cy < 0) cy = 0; else if (cy >= gridH) cy = gridH - 1;
            int cell = cy * gridW + cx;
            particleCell[i] = cell;
            cellCounts[cell]++;
        }

        // Prefix sum to get cell starts
        int sum = 0;
        for (int c = 0; c < numCells; ++c) {
            cellStarts[c] = sum;
            sum += cellCounts[c];
        }
        cellStarts[numCells] = sum; // should be N

        // Prepare write positions
        cellWritePos = cellStarts;

        // Second pass: fill sortedParticles
        for (int i = 0; i < N; ++i) {
            int cell = particleCell[i];
            int &w = cellWritePos[cell];
            sortedParticles[w++] = i;
        }

        // 3. Compute forces and integrate
        const float R = params.cullRadius;
        const float R2 = R * R;
        const int W = gridW, H = gridH;
        auto &cellStart = cellStarts;
        auto &sortedIdx = sortedParticles;
        auto &pCell = particleCell;

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            const Particle pi = particles[i]; // local copy
            Vec2 force(0.0f, 0.0f);

            int cell = pCell[i];
            int cx = cell % W;
            int cy = cell / W;

            // Check neighbor cells within 1 in each direction (since cellSize == cullRadius)
            for (int dy = -1; dy <= 1; ++dy) {
                int ny = cy + dy;
                if (ny < 0 || ny >= H) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    int nx = cx + dx;
                    if (nx < 0 || nx >= W) continue;
                    int nCell = ny * W + nx;
                    int begin = cellStart[nCell];
                    int end = cellStart[nCell + 1];

                    for (int k = begin; k < end; ++k) {
                        int j = sortedIdx[k];
                        if (j == i) continue;

                        // Quick distance squared cull to avoid computeForce call
                        const Vec2 d = particles[j].position - pi.position;
                        const float dist2 = d.length2();
                        if (dist2 <= R2) {
                            force += computeForce(pi, particles[j], R);
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