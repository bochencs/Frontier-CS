#include "world.h"
#include <omp.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

class MySimulator : public Simulator {
private:
    int numThreads = 0;

    // Grid data
    float cellSize = 1.0f;
    float invCellSize = 1.0f;
    float minX = 0.0f, minY = 0.0f;
    int gridW = 1, gridH = 1;

    // Buffers reused across steps
    std::vector<int> particleCell;   // cell index for each particle
    std::vector<int> cellStarts;     // start indices per cell (size = numCells + 1)
    std::vector<int> cellCounts;     // counts per cell (size = numCells)
    std::vector<int> cellIndices;    // particle indices sorted into cells (size = N)

    inline void buildGrid(const std::vector<Particle> &particles, float cullRadius) {
        const int N = (int)particles.size();
        if (N == 0) return;

        // Compute bounding box
        float localMinX = std::numeric_limits<float>::infinity();
        float localMinY = std::numeric_limits<float>::infinity();
        float localMaxX = -std::numeric_limits<float>::infinity();
        float localMaxY = -std::numeric_limits<float>::infinity();

        for (int i = 0; i < N; ++i) {
            const auto &p = particles[i].position;
            if (p.x < localMinX) localMinX = p.x;
            if (p.y < localMinY) localMinY = p.y;
            if (p.x > localMaxX) localMaxX = p.x;
            if (p.y > localMaxY) localMaxY = p.y;
        }

        // Slight padding to avoid boundary issues
        const float pad = 1e-3f;
        localMinX -= pad;
        localMinY -= pad;
        localMaxX += pad;
        localMaxY += pad;

        minX = localMinX;
        minY = localMinY;

        // Grid settings
        cellSize = cullRadius;
        if (cellSize < 1e-6f) cellSize = 1e-6f;
        invCellSize = 1.0f / cellSize;

        int w = std::max(1, (int)std::ceil((localMaxX - localMinX) * invCellSize));
        int h = std::max(1, (int)std::ceil((localMaxY - localMinY) * invCellSize));
        gridW = w;
        gridH = h;
        int numCells = gridW * gridH;

        // Resize buffers
        particleCell.resize(N);
        cellCounts.assign(numCells, 0);
        cellStarts.assign(numCells + 1, 0);
        cellIndices.resize(N);

        // First pass: count per cell
        for (int i = 0; i < N; ++i) {
            const auto &p = particles[i].position;
            int gx = (int)std::floor((p.x - minX) * invCellSize);
            int gy = (int)std::floor((p.y - minY) * invCellSize);
            if (gx < 0) gx = 0;
            if (gy < 0) gy = 0;
            if (gx >= gridW) gx = gridW - 1;
            if (gy >= gridH) gy = gridH - 1;
            int g = gy * gridW + gx;
            particleCell[i] = g;
            cellCounts[g]++;
        }

        // Prefix sums to get starts
        int sum = 0;
        for (int c = 0; c < numCells; ++c) {
            cellStarts[c] = sum;
            sum += cellCounts[c];
        }
        cellStarts[numCells] = sum;

        // Temp offsets to write particles into cells
        std::vector<int> offsets(numCells, 0);
        for (int c = 0; c < numCells; ++c) {
            offsets[c] = cellStarts[c];
        }

        // Second pass: fill indices
        for (int i = 0; i < N; ++i) {
            int g = particleCell[i];
            int idx = offsets[g]++;
            cellIndices[idx] = i;
        }
    }

public:
    void init(int numParticles, StepParameters params) override {
        numThreads = omp_get_max_threads();
        omp_set_num_threads(numThreads);
        particleCell.resize(numParticles);
        cellIndices.resize(numParticles);
        cellStarts.clear();
        cellCounts.clear();
    }

    void simulateStep(std::vector<Particle> &particles,
                      std::vector<Particle> &newParticles,
                      StepParameters params) override {
        const int N = (int)particles.size();
        if ((int)newParticles.size() != N) newParticles.resize(N);
        if (N == 0) return;

        // Build uniform grid
        buildGrid(particles, params.cullRadius);

        const float r = params.cullRadius;
        const float r2 = r * r;

        // Parallel compute forces using spatial grid
        #pragma omp parallel for schedule(dynamic, 32)
        for (int i = 0; i < N; ++i) {
            const Particle pi = particles[i];
            Vec2 force(0.0f, 0.0f);

            int cell = particleCell[i];
            int gy = cell / gridW;
            int gx = cell - gy * gridW;

            for (int dy = -1; dy <= 1; ++dy) {
                int ny = gy + dy;
                if (ny < 0 || ny >= gridH) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    int nx = gx + dx;
                    if (nx < 0 || nx >= gridW) continue;
                    int ncell = ny * gridW + nx;

                    int start = cellStarts[ncell];
                    int end = cellStarts[ncell + 1];
                    for (int idx = start; idx < end; ++idx) {
                        int j = cellIndices[idx];
                        if (j == i) continue;
                        const Vec2 d = particles[j].position - pi.position;
                        float dist2 = d.x * d.x + d.y * d.y;
                        if (dist2 <= r2 && dist2 >= 1e-12f) {
                            force += computeForce(pi, particles[j], r);
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