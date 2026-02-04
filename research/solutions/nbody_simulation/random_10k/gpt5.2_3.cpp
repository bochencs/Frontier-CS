#include "world.h"
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

class MySimulator : public Simulator {
private:
    int numThreads = 16;

    float cellSize = 0.0f;
    float invCellSize = 0.0f;
    float originX = 0.0f, originY = 0.0f;
    int dimX = 0, dimY = 0, numCells = 0;
    int rCells = 0;

    std::vector<int> cellCount;
    std::vector<int> cellStart;      // size numCells+1
    std::vector<int> cellWrite;      // size numCells
    std::vector<int> cellParticles;  // size N
    std::vector<int> cellOfParticle; // size N

    static inline void addForceFast(float &fx, float &fy,
                                    float dx, float dy, float d2,
                                    float pimG, float pjm,
                                    float cullRadius,
                                    float decayStart, float invDecayDen) {
        float dist = std::sqrt(d2);
        if (dist < 1e-3f) return;
        if (dist > cullRadius) return;

        float invDist = 1.0f / dist;
        float dirx = dx * invDist;
        float diry = dy * invDist;

        float distAdj = (dist < 1e-1f) ? 1e-1f : dist;
        float invDistAdj2 = 1.0f / (distAdj * distAdj);

        float f = pimG * pjm * invDistAdj2;

        if (distAdj > decayStart) {
            float decay = 1.0f - (distAdj - decayStart) * invDecayDen;
            f *= decay;
        }

        fx += dirx * f;
        fy += diry * f;
    }

    inline void buildGridSequential(const std::vector<Particle> &particles, float cullRadius) {
        const int N = (int)particles.size();
        if ((int)cellParticles.size() != N) cellParticles.resize(N);
        if ((int)cellOfParticle.size() != N) cellOfParticle.resize(N);

        float minx = particles[0].position.x, maxx = minx;
        float miny = particles[0].position.y, maxy = miny;
        for (int i = 1; i < N; ++i) {
            float x = particles[i].position.x;
            float y = particles[i].position.y;
            minx = std::min(minx, x);
            maxx = std::max(maxx, x);
            miny = std::min(miny, y);
            maxy = std::max(maxy, y);
        }

        cellSize = std::max(1e-3f, cullRadius * 0.2f);
        invCellSize = 1.0f / cellSize;

        float margin = cullRadius + 1e-3f;
        originX = minx - margin;
        originY = miny - margin;

        float w = (maxx - minx) + 2.0f * margin;
        float h = (maxy - miny) + 2.0f * margin;
        dimX = (int)(w * invCellSize) + 1;
        dimY = (int)(h * invCellSize) + 1;
        if (dimX < 1) dimX = 1;
        if (dimY < 1) dimY = 1;

        long long cellsLL = 1LL * dimX * dimY;
        if (cellsLL < 1) cellsLL = 1;
        if (cellsLL > (1LL << 20)) { // fallback handled by caller
            numCells = (int)std::min(cellsLL, (long long)(1LL << 20));
            return;
        }
        numCells = (int)cellsLL;

        if ((int)cellCount.size() != numCells) cellCount.assign(numCells, 0);
        else std::fill(cellCount.begin(), cellCount.end(), 0);

        // Count
        for (int i = 0; i < N; ++i) {
            float x = particles[i].position.x;
            float y = particles[i].position.y;
            int ix = (int)((x - originX) * invCellSize);
            int iy = (int)((y - originY) * invCellSize);
            if (ix < 0) ix = 0;
            else if (ix >= dimX) ix = dimX - 1;
            if (iy < 0) iy = 0;
            else if (iy >= dimY) iy = dimY - 1;
            int c = ix + iy * dimX;
            cellOfParticle[i] = c;
            cellCount[c]++;
        }

        if ((int)cellStart.size() != numCells + 1) cellStart.resize(numCells + 1);
        if ((int)cellWrite.size() != numCells) cellWrite.resize(numCells);

        int sum = 0;
        for (int c = 0; c < numCells; ++c) {
            cellStart[c] = sum;
            sum += cellCount[c];
        }
        cellStart[numCells] = sum;

        // Deterministic fill (stable by particle index)
        for (int c = 0; c < numCells; ++c) cellWrite[c] = cellStart[c];
        for (int i = 0; i < N; ++i) {
            int c = cellOfParticle[i];
            int idx = cellWrite[c]++;
            cellParticles[idx] = i;
        }

        rCells = (int)std::ceil(cullRadius / cellSize);
        if (rCells < 1) rCells = 1;
    }

public:
    void init(int /*numParticles*/, StepParameters /*params*/) override {
        int procs = omp_get_num_procs();
        numThreads = std::min(16, std::max(1, procs));
        omp_set_dynamic(0);
        omp_set_num_threads(numThreads);

        cellCount.clear();
        cellStart.clear();
        cellWrite.clear();
        cellParticles.clear();
        cellOfParticle.clear();

        dimX = dimY = numCells = 0;
        rCells = 0;
        cellSize = invCellSize = 0.0f;
        originX = originY = 0.0f;
    }

    void simulateStep(std::vector<Particle> &particles,
                      std::vector<Particle> &newParticles,
                      StepParameters params) override {
        const int N = (int)particles.size();
        if (N == 0) return;

        const float cullRadius = params.cullRadius;
        const float cull2 = cullRadius * cullRadius;
        const float cull2Slack = cull2 * 1.0001f;

        // Build grid deterministically; if grid would be too large, fall back.
        // (In typical benchmark settings, grid is small.)
        buildGridSequential(particles, cullRadius);
        long long cellsLL = 1LL * dimX * dimY;
        const bool fallbackBrute = (cellsLL > (1LL << 20));

        const float decayStart = cullRadius * 0.75f;
        const float invDecayDen = 4.0f / cullRadius; // 1/(cullRadius*0.25)
        const float deltaTime = params.deltaTime;

        if (fallbackBrute) {
            #pragma omp parallel for schedule(dynamic, 32)
            for (int i = 0; i < N; ++i) {
                const Particle pi = particles[i];
                const float pix = pi.position.x;
                const float piy = pi.position.y;
                const float pimG = pi.mass * 0.01f;

                float fx = 0.0f, fy = 0.0f;
                for (int j = 0; j < N; ++j) {
                    if (j == i) continue;
                    const Particle &pj = particles[j];
                    float dx = pj.position.x - pix;
                    float dy = pj.position.y - piy;
                    float d2 = dx * dx + dy * dy;
                    if (d2 < 1e-6f) continue;
                    if (d2 > cull2Slack) continue;
                    addForceFast(fx, fy, dx, dy, d2, pimG, pj.mass, cullRadius, decayStart, invDecayDen);
                }
                newParticles[i] = updateParticle(pi, Vec2(fx, fy), deltaTime);
            }
            return;
        }

        #pragma omp parallel for schedule(dynamic, 32)
        for (int i = 0; i < N; ++i) {
            const Particle pi = particles[i];
            const float pix = pi.position.x;
            const float piy = pi.position.y;
            const float pimG = pi.mass * 0.01f;

            float fx = 0.0f, fy = 0.0f;

            int cell = cellOfParticle[i];
            int ix = cell - (cell / dimX) * dimX;
            int iy = cell / dimX;

            int x0 = ix - rCells;
            int x1 = ix + rCells;
            int y0 = iy - rCells;
            int y1 = iy + rCells;
            if (x0 < 0) x0 = 0;
            if (y0 < 0) y0 = 0;
            if (x1 >= dimX) x1 = dimX - 1;
            if (y1 >= dimY) y1 = dimY - 1;

            for (int yy = y0; yy <= y1; ++yy) {
                int row = yy * dimX;
                for (int xx = x0; xx <= x1; ++xx) {
                    int c2 = row + xx;
                    int start = cellStart[c2];
                    int end = cellStart[c2 + 1];
                    for (int k = start; k < end; ++k) {
                        int j = cellParticles[k];
                        if (j == i) continue;
                        const Particle &pj = particles[j];
                        float dx = pj.position.x - pix;
                        float dy = pj.position.y - piy;
                        float d2 = dx * dx + dy * dy;
                        if (d2 < 1e-6f) continue;
                        if (d2 > cull2Slack) continue;
                        addForceFast(fx, fy, dx, dy, d2, pimG, pj.mass, cullRadius, decayStart, invDecayDen);
                    }
                }
            }

            newParticles[i] = updateParticle(pi, Vec2(fx, fy), deltaTime);
        }
    }
};

Simulator* createSimulator() {
    return new MySimulator();
}