#include<iostream>
#include<vector>
#include<fstream>
#include<cmath>
#include<algorithm>
#include<string>
#include<limits>
#include<cstdlib>
#include<omp.h>
#include<unordered_map>
#include<chrono>
#include<cuda_runtime.h>
using namespace std::chrono;

struct Point {
    float x, y, z;
    int intesity;
};
__global__ void knn_cuda(
    float *x, float *y, float *z,
    int *intensity, int *out_intensity,
    int n, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int KMAX = 130;

    float best_dist[KMAX];
    int b_idx[KMAX];

    int curr_k = 0;

    for (int t = 0; t < KMAX; t++) {
        best_dist[t] = 1e30;
        b_idx[t] = -1;
    }

    for (int j = 0; j < n; j++) {
        float dx = x[i] - x[j];
        float dy = y[i] - y[j];
        float dz = z[i] - z[j];
        float d = dx*dx + dy*dy + dz*dz;

        if (curr_k < k+1) {
            best_dist[curr_k] = d;
            b_idx[curr_k] = j;
            curr_k++;
        } else {
            int worst = 0;
            for (int t = 1; t < k+1; t++) {

                int a = b_idx[t];
                int b = b_idx[worst];

                if (best_dist[t] > best_dist[worst] ||
                    (best_dist[t] == best_dist[worst] &&
                    (x[a] > x[b] ||
                    (x[a] == x[b] && y[a] > y[b]) ||
                    (x[a] == x[b] && y[a] == y[b] && z[a] > z[b]))))
                {
                    worst = t;
                }
            }

            int w = worst;
            int a = j;
            int b = b_idx[w];

            if (d < best_dist[w] ||
            (d == best_dist[w] &&
                (x[a] < x[b] ||
                (x[a] == x[b] && y[a] < y[b]) ||
                (x[a] == x[b] && y[a] == y[b] && z[a] < z[b]))))
            {
                best_dist[w] = d;
                b_idx[w] = j;
            }
        }
    }

    int H[256];
    for (int v = 0; v < 256; v++) H[v] = 0;

    for (int t = 0; t < k+1; t++) {
        int j = b_idx[t];
        if (j >= 0)
            H[intensity[j]]++;
    }

    int C[256];
    C[0] = H[0];
    for (int v = 1; v < 256; v++)
        C[v] = C[v-1] + H[v];

    int Ci_min = 0;
    for (int v = 0; v < 256; v++) {
        if (C[v] > 0) {
            Ci_min = C[v];
            break;
        }
    }

    int m = k + 1;

    if (m == Ci_min) {
        out_intensity[i] = intensity[i];
        return;
    }

    float v = (((float)(C[intensity[i]] - Ci_min) / ((float)(m - Ci_min))) * (255.0f));
    int updated_i = (int)floor(v);

    if (updated_i < 0) updated_i = 0;
    if (updated_i > 255) updated_i = 255;

    out_intensity[i] = updated_i;
    
}


void process_approx_knn(std::vector<Point>& pts, int k) {
    int n = pts.size();

    float minx = 1e18, miny = 1e18, minz = 1e18;
    float maxx = -1e18, maxy = -1e18, maxz = -1e18;

    for (int i = 0; i < n; i++) {
        minx = min(minx, pts[i].x);
        miny = min(miny, pts[i].y);
        minz = min(minz, pts[i].z);

        maxx = max(maxx, pts[i].x);
        maxy = max(maxy, pts[i].y);
        maxz = max(maxz, pts[i].z);
    }

    int grid_dim = max(1, (int)cbrt(n));

    float cell_x = (maxx - minx) / grid_dim + 1e-9;
    float cell_y = (maxy - miny) / grid_dim + 1e-9;
    float cell_z = (maxz - minz) / grid_dim + 1e-9;

    std::unordered_map<long long, std::vector<int>> grid;

    auto hash = [&](int gx, int gy, int gz) {
        return ((long long)gx << 40) | ((long long)gy << 20) | gz;
    };

    for (int i = 0; i < n; i++) {
        int gx = (int)(((float)(pts[i].x - minx)) / (float)cell_x);
        int gy = (int)(((float)(pts[i].y - miny)) / (float)cell_y);
        int gz = (int)(((float)(pts[i].z - minz)) / (float)cell_z);
        grid[hash(gx, gy, gz)].push_back(i);
    }

    std::vector<int> out(n);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {

        int gx = (int)(((float)(pts[i].x - minx)) / (float)cell_x);
        int gy = (int)(((float)(pts[i].y - miny)) / (float)cell_y);
        int gz = (int)(((float)(pts[i].z - minz)) / (float)cell_z);

        std::vector<std::pair<float,int>> candidates;

        int radius = 0;

        while (true) {
            candidates.clear();

            for (int dx = -radius; dx <= radius; dx++) {
                for (int dy = -radius; dy <= radius; dy++) {
                    for (int dz = -radius; dz <= radius; dz++) {

                        long long key = hash(gx+dx, gy+dy, gz+dz);
                        auto it = grid.find(key);
                        if (it == grid.end()) continue;

                        for (int j : it->second) {
                            if (j == i) continue;

                            float dx_ = pts[i].x - pts[j].x;
                            float dy_ = pts[i].y - pts[j].y;
                            float dz_ = pts[i].z - pts[j].z;
                            float d = dx_*dx_ + dy_*dy_ + dz_*dz_;

                            candidates.push_back({d, j});
                        }
                    }
                }
            }

            if ((int)candidates.size() >= k || radius > grid_dim) break;

            radius++;
        }

        int take = min(k, (int)candidates.size());

        if (candidates.empty()) {
            out[i] = pts[i].intesity;
            continue;
        }

        partial_sort(
            candidates.begin(),
            candidates.begin() + take,
            candidates.end(),
            [&](const std::pair<float,int>& a, const std::pair<float,int>& b) {

                if (a.first != b.first) return a.first < b.first;

                const Point &pa = pts[a.second];
                const Point &pb = pts[b.second];

                if (pa.x != pb.x) return pa.x < pb.x;
                if (pa.y != pb.y) return pa.y < pb.y;
                return pa.z < pb.z;
            }
        );

        int H[256] = {0};

        H[pts[i].intesity]++;

        for (int t = 0; t < take; t++) {
            H[pts[candidates[t].second].intesity]++;
        }

        int C[256];
        C[0] = H[0];
        for (int v = 1; v < 256; v++)
            C[v] = C[v-1] + H[v];

        int Ci_min = 0;
        for (int v = 0; v < 256; v++) {
            if (C[v] > 0) {
                Ci_min = C[v];
                break;
            }
        }

        int m = take + 1;

        if (m == Ci_min) {
            out[i] = pts[i].intesity;
            continue;
        }

        float v = (((float)(C[pts[i].intesity] - Ci_min) / (float)(m - Ci_min))* (255.0f));
        int updated_i = (int)floor(v);

        if (updated_i < 0) updated_i = 0;
        if (updated_i > 255) updated_i = 255;

        out[i] = updated_i;
    }

    std::ofstream fout("approx_knn.txt");
    for (int i = 0; i < n; i++) {
        fout << pts[i].x << " "
             << pts[i].y << " "
             << pts[i].z << " "
             << out[i] << "\n";
    }
}

void kmeans(std::vector<Point>& pts, int k, int T, std::vector<int>& labels) {
    int n = pts.size();

    std::vector<Point> centroids(k);
    for (int i = 0; i < k; i++)
        centroids[i] = pts[i];

    labels.assign(n, 0);

    std::vector<int> old_labels(n, -1);
    int iterations;
    for (iterations = 0; iterations < T; iterations++) {
        std::vector<int> old_labels = labels;
        for (int i = 0; i < n; i++) {
            int b_id = 0;
            float best = 1e30;

            for (int c = 0; c < k; c++) {
                float dx = pts[i].x - centroids[c].x;
                float dy = pts[i].y - centroids[c].y;
                float dz = pts[i].z - centroids[c].z;
                float d = dx*dx + dy*dy + dz*dz;

                if (d < best ||
                (d == best &&
                    (pts[i].x < centroids[c].x ||
                    (pts[i].x == centroids[c].x && pts[i].y < centroids[c].y) ||
                    (pts[i].x == centroids[c].x && pts[i].y == centroids[c].y && pts[i].z < centroids[c].z))))
                {
                    best = d;
                    b_id = c;
                }
            }

            labels[i] = b_id;
        }

        std::vector<float> sx(k, 0), sy(k, 0), sz(k, 0);
        std::vector<int> cnt(k, 0);

        #pragma omp parallel
        {
            std::vector<float> sx_local(k, 0), sy_local(k, 0), sz_local(k, 0);
            std::vector<int> cnt_local(k, 0);

            #pragma omp for nowait
            for (int i = 0; i < n; i++) {
                int c = labels[i];
                sx_local[c] += pts[i].x;
                sy_local[c] += pts[i].y;
                sz_local[c] += pts[i].z;
                cnt_local[c]++;
            }

            #pragma omp critical
            {
                for (int c = 0; c < k; c++) {
                    sx[c] += sx_local[c];
                    sy[c] += sy_local[c];
                    sz[c] += sz_local[c];
                    cnt[c] += cnt_local[c];
                }
            }
        }

        for (int c = 0; c < k; c++) {
            if (cnt[c] > 0) {
                centroids[c].x = (int)(sx[c] / cnt[c]);
                centroids[c].y = (int)(sy[c] / cnt[c]);
                centroids[c].z = (int)(sz[c] / cnt[c]);
            }
        }

        bool changed = false;
        for (int i = 0; i < n; i++) {
            if (labels[i] != old_labels[i]) {
                changed = true;
                break;
            }
        }

        if (!changed) break;
    }
}

void process_kmeans(std::vector<Point>& pts, int k, int T) {
    int n = pts.size();

    std::vector<int> labels;
    kmeans(pts, k, T, labels);

    std::vector<std::vector<int>> clusters(k);
    for (int i = 0; i < n; i++) {
        clusters[labels[i]].push_back(i);
    }

    std::vector<int> out(n);

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {

        int cid = labels[i];
        auto &cluster = clusters[cid];

        int H[256] = {0};

        for (int j : cluster) {
            H[pts[j].intesity]++;
        }
        int C[256];
        C[0] = H[0];
        for (int v = 1; v < 256; v++)
            C[v] = C[v-1] + H[v];

        int Ci_min = 0;
        for (int v = 0; v < 256; v++) {
            if (C[v] > 0) {
                Ci_min = C[v];
                break;
            }
        }

        int m = cluster.size();

        if (m == Ci_min) {
            out[i] = pts[i].intesity;
            continue;
        }

        float v = (((float)(C[pts[i].intesity] - Ci_min) / ((float)(m - Ci_min))) * (255.0f));
        int updated_i = (int)floor(v);

        if (updated_i < 0) updated_i = 0;
        if (updated_i > 255) updated_i = 255;

        out[i] = updated_i;
    }

    std::ofstream fout("kmeans.txt");
    for (int i = 0; i < n; i++) {
        fout << pts[i].x << " "
             << pts[i].y << " "
             << pts[i].z << " "
             << out[i] << "\n";
    }
}


int main(int argc, char** argv) {
    std::string file = argv[1];
    std::string mode = argv[2];

    std::ifstream fin(file);
    int n, k, T;
    fin >> n >> k >> T;

    std::vector<Point> pts(n);
    for (int i = 0; i < n; i++)
        fin >> pts[i].x >> pts[i].y >> pts[i].z >> pts[i].intesity;

    float *dx, *dy, *dz;
    int *dI, *dOut;

    cudaMalloc(&dx, n*sizeof(float));
    cudaMalloc(&dy, n*sizeof(float));
    cudaMalloc(&dz, n*sizeof(float));
    cudaMalloc(&dI, n*sizeof(int));
    cudaMalloc(&dOut, n*sizeof(int));

    std::vector<float> hx(n), hy(n), hz(n);
    std::vector<int> hI(n);

    for (int i = 0; i < n; i++) {
        hx[i] = pts[i].x;
        hy[i] = pts[i].y;
        hz[i] = pts[i].z;
        hI[i] = pts[i].intesity;
    }

    cudaMemcpy(dx, hx.data(), n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy.data(), n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dz, hz.data(), n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dI, hI.data(), n*sizeof(int), cudaMemcpyHostToDevice);

    
    int block = 256;
    int grid = (n + block - 1) / block;

    if (mode == "knn") {

        auto total_start = high_resolution_clock::now();
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);


        cudaEventRecord(start);

        knn_cuda<<<grid, block>>>(
            dx, dy, dz, dI, dOut, n, k);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);

        std::vector<int> hOut(n);
        cudaMemcpy(hOut.data(), dOut, n*sizeof(int), cudaMemcpyDeviceToHost);

        std::ofstream fout("knn.txt");
        for (int i = 0; i < n; i++) {
            fout << pts[i].x << " "
                << pts[i].y << " "
                << pts[i].z << " "
                << hOut[i] << "\n";
        }

        auto total_end = high_resolution_clock::now();
        double total_time = duration<double>(total_end - total_start).count();
        std::cout << "KNN Kernel Time: " << ms / 1000.0 << " seconds\n";
        std::cout << "KNN Total Time : " << total_time << " seconds\n";
    }
    else if (mode == "approx_knn") {
        auto start = high_resolution_clock::now();
        process_approx_knn(pts, k);
        auto end = high_resolution_clock::now();
        double t = duration<double>(end - start).count();
        std::cout << "Approx KNN Time: " << t << " seconds\n";
    }
    else if (mode == "kmeans") {
        auto start = high_resolution_clock::now();
        process_kmeans(pts, k, T);
        auto end = high_resolution_clock::now();
        double t = duration<double>(end - start).count();
        std::cout << "KMeans Time: " << t << " seconds\n";
    }

    cudaFree(dx); cudaFree(dy); cudaFree(dz);
    cudaFree(dI); cudaFree(dOut);

    return 0;
}