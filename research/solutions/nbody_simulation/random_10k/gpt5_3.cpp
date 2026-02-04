#include "world.h"
#include <omp.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <cstdint>

class MySimulator : public Simulator {
private:
    int numThreads = 0;

    // Grid data
    float cellSize = 1.0f;
    int gridX = 1, gridY = 1;
    float minX = 0.0f, minY = 0.0f;

    // Reusable buffers
    std::vector<int> cellCounts;
    std::vector<int> cellStart;   // size = numCells + 1 (exclusive prefix sum)
    std::vector<int> cellWrite;   // temp, same size as cellStart (without +1)
    std::vector<int> particleCell; // per-particle linear cell index
    std::vector<int> sortedIndices; // particles sorted into cells

    // Helper to compute grid dimensions and cell size based on bounding box
    void computeGridParams(const std::vector<Particle>& particles, float cullRadius) {
        if (particles.empty()) {
            gridX = gridY = 1;
            cellSize = std::max(cullRadius, 1e-3f);
            minX = minY = 0.0f;
            return;
        }

        // Compute bounding box
        float localMinX = particles[0].position.x;
        float localMaxX = particles[0].position.x;
        float localMinY = particles[0].position.y;
        float localMaxY = particles[0].position.y;

        for (size_t i = 1; i < particles.size(); ++i) {
            float x = particles[i].position.x;
            float y = particles[i].position.y;
            if (x < localMinX) localMinX = x;
            if (x > localMaxX) localMaxX = x;
            if (y < localMinY) localMinY = y;
            if (y > localMaxY) localMaxY = y;
        }

        minX = localMinX;
        minY = localMinY;
        float width = localMaxX - localMinX;
        float height = localMaxY - localMinY;

        // Preferred small cell size for efficiency; fallback to larger if too many cells
        float preferred = std::max(cullRadius / 4.0f, 1e-3f);
        cellSize = preferred;

        gridX = std::max(1, (int)std::floor(width / cellSize) + 1);
        gridY = std::max(1, (int)std::floor(height / cellSize) + 1);

        // If grid too large, increase cell size to reduce memory usage
        // Cap total cells relative to particle count for safety
        size_t numCells = (size_t)gridX * (size_t)gridY;
        size_t N = particles.size();
        const size_t maxCells = std::max<size_t>(65536, 4 * N + 1024); // adaptive cap
        if (numCells > maxCells) {
            cellSize = std::max(cullRadius, 1e-3f);
            gridX = std::max(1, (int)std::floor(width / cellSize) + 1);
            gridY = std::max(1, (int)std::floor(height / cellSize) + 1);
        }
    }

    inline int clampIndex(int v, int lo, int hi) const {
        if (v < lo) return lo;
        if (v > hi) return hi;
        return v;
    }

public:
    void init(int numParticles, StepParameters params) override {
        numThreads = omp_get_max_threads();
        omp_set_num_threads(numThreads);

        // Reserve buffers to avoid reallocations
        cellCounts.clear();
        cellStart.clear();
        cellWrite.clear();
        particleCell.clear();
        sortedIndices.clear();

        particleCell.reserve((size_t)numParticles);
        sortedIndices.reserve((size_t)numParticles);
    }

    void simulateStep(std::vector<Particle> &particles,
                      std::vector<Particle> &newParticles,
                      StepParameters params) override {
        const size_t N = particles.size();
        if (N == 0) return;
        if (newParticles.size() != N) newParticles.resize(N);

        // Compute grid parameters for this step
        computeGridParams(particles, params.cullRadius);

        const size_t numCells = (size_t)gridX * (size_t)gridY;
        cellCounts.assign(numCells, 0);
        particleCell.resize(N);

        // Assign particles to cells and count
        for (size_t i = 0; i < N; ++i) {
            float x = particles[i].position.x;
            float y = particles[i].position.y;
            int cx = (int)std::floor((x - minX) / cellSize);
            int cy = (int)std::floor((y - minY) / cellSize);
            cx = clampIndex(cx, 0, gridX - 1);
            cy = clampIndex(cy, 0, gridY - 1);
            int cell = cy * gridX + cx;
            particleCell[i] = cell;
            cellCounts[(size_t)cell]++;
        }

        // Exclusive prefix sum to compute cell starts
        cellStart.resize(numCells + 1);
        cellStart[0] = 0;
        for (size_t c = 0; c < numCells; ++c) {
            cellStart[c + 1] = cellStart[c] + cellCounts[c];
        }

        // Fill particles sorted by cell
        sortedIndices.resize(N);
        cellWrite = cellStart; // copy start offsets to write positions
        for (size_t i = 0; i < N; ++i) {
            int cell = particleCell[i];
            int pos = cellWrite[(size_t)cell]++;
            sortedIndices[(size_t)pos] = (int)i;
        }

        // Neighbor range in cells
        const int range = (int)std::ceil(params.cullRadius / cellSize);
        const float cull = params.cullRadius;

        // Parallel compute
        const Particle* pData = particles.data();
        Particle* nData = newParticles.data();
        const int* sIdx = sortedIndices.data();
        const int* cStart = cellStart.data();

        #pragma omp parallel for schedule(static)
        for (int ii = 0; ii < (int)N; ++ii) {
            const Particle pi = pData[ii];
            Vec2 totalForce(0.0f, 0.0f);

            int baseCell = particleCell[(size_t)ii];
            int cY = baseCell / gridX;
            int cX = baseCell - cY * gridX;

            int y0 = cY - range;
            int y1 = cY + range;
            if (y0 < 0) y0 = 0;
            if (y1 >= gridY) y1 = gridY - 1;

            for (int yy = y0; yy <= y1; ++yy) {
                int x0 = cX - range;
                int x1 = cX + range;
                if (x0 < 0) x0 = 0;
                if (x1 >= gridX) x1 = gridX - 1;

                int rowBase = yy * gridX;
                for (int xx = x0; xx <= x1; ++xx) {
                    int cell = rowBase + xx;
                    int begin = cStart[(size_t)cell];
                    int end   = cStart[(size_t)cell + 1];
                    for (int idx = begin; idx < end; ++idx) {
                        int j = sIdx[(size_t)idx];
                        if (j == ii) continue;
                        // Pre-check to mirror baseline behavior (< cullRadius) using same length() op
                        Vec2 diff = pi.position - pData[j].position;
                        if (diff.length() < cull) {
                            totalForce += computeForce(pi, pData[j], cull);
                        }
                    }
                }
            }

            nData[ii] = updateParticle(pi, totalForce, params.deltaTime);
        }
    }
};

Simulator* createSimulator() {
    return new MySimulator();
}