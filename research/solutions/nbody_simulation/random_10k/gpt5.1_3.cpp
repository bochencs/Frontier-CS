#include "world.h"
#include <omp.h>
#include <vector>
#include <cmath>
#include <algorithm>

class MySimulator : public Simulator {
private:
    int numThreads;
    float cellSize;
    float invCellSize;
    float originX;
    float originY;
    int gridWidth;
    int gridHeight;
    std::vector<int> cellStart;       // Start index in particleIndices for each cell
    std::vector<int> cellCount;       // Number of particles in each cell
    std::vector<int> particleIndices; // Indices of particles sorted into cells

    inline void buildGrid(const std::vector<Particle> &particles,
                          float cullRadius) {
        int N = (int)particles.size();
        if (N == 0) {
            gridWidth = gridHeight = 0;
            cellStart.clear();
            cellCount.clear();
            particleIndices.clear();
            return;
        }

        // Compute bounding box
        float xmin = particles[0].position.x;
        float xmax = particles[0].position.x;
        float ymin = particles[0].position.y;
        float ymax = particles[0].position.y;

        for (int i = 1; i < N; ++i) {
            float x = particles[i].position.x;
            float y = particles[i].position.y;
            if (x < xmin) xmin = x;
            if (x > xmax) xmax = x;
            if (y < ymin) ymin = y;
            if (y > ymax) ymax = y;
        }

        // Add small padding to avoid degenerate dimensions
        const float padding = 1e-3f;
        xmin -= padding;
        ymin -= padding;
        xmax += padding;
        ymax += padding;

        float width = xmax - xmin;
        float height = ymax - ymin;
        if (width <= 0.0f) width = 1.0f;
        if (height <= 0.0f) height = 1.0f;
        float area = width * height;

        // Choose cell size based on cullRadius and world area, with cap on cell count
        const float targetMaxCells = 40000.0f; // approximate cap on number of cells
        float minCellSizeFromArea = std::sqrt(area / targetMaxCells);
        float baseCellSize = cullRadius / 3.0f; // reasonably fine grid
        if (baseCellSize < 1.0f) baseCellSize = 1.0f;
        cellSize = std::max(baseCellSize, minCellSizeFromArea);
        invCellSize = 1.0f / cellSize;

        gridWidth = std::max(1, (int)std::floor(width * invCellSize) + 1);
        gridHeight = std::max(1, (int)std::floor(height * invCellSize) + 1);
        int numCells = gridWidth * gridHeight;

        if ((int)cellStart.size() != numCells) {
            cellStart.resize(numCells);
            cellCount.resize(numCells);
        }
        std::fill(cellCount.begin(), cellCount.end(), 0);

        if ((int)particleIndices.size() != N) {
            particleIndices.resize(N);
        }

        originX = xmin;
        originY = ymin;

        // First pass: count particles per cell
        for (int i = 0; i < N; ++i) {
            float x = particles[i].position.x;
            float y = particles[i].position.y;
            int cx = (int)((x - originX) * invCellSize);
            int cy = (int)((y - originY) * invCellSize);
            if (cx < 0) cx = 0;
            else if (cx >= gridWidth) cx = gridWidth - 1;
            if (cy < 0) cy = 0;
            else if (cy >= gridHeight) cy = gridHeight - 1;
            int cellIndex = cy * gridWidth + cx;
            ++cellCount[cellIndex];
        }

        // Prefix sum to compute cellStart
        int sum = 0;
        for (int c = 0; c < numCells; ++c) {
            cellStart[c] = sum;
            sum += cellCount[c];
            // Reuse cellCount as running offset during the next pass
            cellCount[c] = 0;
        }

        // Second pass: fill particleIndices
        for (int i = 0; i < N; ++i) {
            float x = particles[i].position.x;
            float y = particles[i].position.y;
            int cx = (int)((x - originX) * invCellSize);
            int cy = (int)((y - originY) * invCellSize);
            if (cx < 0) cx = 0;
            else if (cx >= gridWidth) cx = gridWidth - 1;
            if (cy < 0) cy = 0;
            else if (cy >= gridHeight) cy = gridHeight - 1;
            int cellIndex = cy * gridWidth + cx;
            int destIndex = cellStart[cellIndex] + cellCount[cellIndex]++;
            particleIndices[destIndex] = i;
        }
        // After this, cellCount[c] holds the final count for cell c
    }

public:
    MySimulator()
        : numThreads(1),
          cellSize(1.0f),
          invCellSize(1.0f),
          originX(0.0f),
          originY(0.0f),
          gridWidth(0),
          gridHeight(0) {}

    void init(int numParticles, StepParameters params) override {
        numThreads = omp_get_max_threads();
        if (numThreads <= 0) numThreads = 1;
        omp_set_num_threads(numThreads);

        // Reserve for particle indices to avoid repeated allocations
        particleIndices.reserve(numParticles);
    }

    void simulateStep(std::vector<Particle> &particles,
                      std::vector<Particle> &newParticles,
                      StepParameters params) override {
        int N = (int)particles.size();
        if (N == 0) return;
        if ((int)newParticles.size() != N) {
            newParticles.resize(N);
        }

        float cullRadius = params.cullRadius;

        // Build spatial grid for this step
        buildGrid(particles, cullRadius);

        int gridW = gridWidth;
        int gridH = gridHeight;
        float invCell = invCellSize;
        float originXLocal = originX;
        float originYLocal = originY;

        int *cellStartPtr = cellStart.data();
        int *cellCountPtr = cellCount.data();
        int *particleIndicesPtr = particleIndices.data();

        // Maximum cell search range in each dimension
        int maxRange = (int)std::ceil(cullRadius * invCell);
        if (maxRange < 0) maxRange = 0;

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            const Particle &pi = particles[i];
            Vec2 force = Vec2(0.0f, 0.0f);

            int cx = (int)((pi.position.x - originXLocal) * invCell);
            int cy = (int)((pi.position.y - originYLocal) * invCell);
            if (cx < 0) cx = 0;
            else if (cx >= gridW) cx = gridW - 1;
            if (cy < 0) cy = 0;
            else if (cy >= gridH) cy = gridH - 1;

            for (int dy = -maxRange; dy <= maxRange; ++dy) {
                int ny = cy + dy;
                if (ny < 0 || ny >= gridH) continue;
                int rowOffset = ny * gridW;
                for (int dx = -maxRange; dx <= maxRange; ++dx) {
                    int nx = cx + dx;
                    if (nx < 0 || nx >= gridW) continue;
                    int cellIndex = rowOffset + nx;
                    int start = cellStartPtr[cellIndex];
                    int count = cellCountPtr[cellIndex];
                    int end = start + count;
                    for (int idx = start; idx < end; ++idx) {
                        int j = particleIndicesPtr[idx];
                        if (j == i) continue;
                        force += computeForce(pi, particles[j], cullRadius);
                    }
                }
            }

            newParticles[i] = updateParticle(pi, force, params.deltaTime);
        }
    }
};

// Factory function
Simulator* createSimulator() {
    return new MySimulator();
}