#include <iostream>
#include <vector>
#include <bitset>
#include <algorithm>
#include <numeric>
#include <chrono> 
#include <mpi.h>
#include <fstream>
using namespace std;

#define pb push_back
#define all(x) (x).begin(), (x).end()
#define vi vector<int>

const int MAXN = 1200;
struct Vertex {
    int profit, cost;
};

int N, E, B;
vector<Vertex> vertices;
vector<bitset<MAXN>> adj;
vi ratio_order, global_order;
int local_best = 0;
vi local_best_clique;

void broadcast_graph(int rank, int size) {
    for (int i = 0; i < N; ++i) {
        unsigned long long chunks[MAXN / 64 + 1] = {0};
        if (rank == 0) {
            for (int j = 0; j < MAXN; ++j) {
                if (adj[i].test(j)) chunks[j / 64] |= (1ULL << (j % 64));
            }
        }
        MPI_Bcast(chunks, MAXN / 64 + 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            for (int j = 0; j < MAXN; ++j) {
                if (chunks[j / 64] & (1ULL << (j % 64))) adj[i].set(j);
            }
        }
    }
}


int StructuralBound(bitset<MAXN> cand) {
    int bound = 0;
    int numColors = 0;
    static bitset<MAXN> colorSets[MAXN];
    static int colorMax[MAXN];
    for(int i=0; i<numColors; ++i) colorSets[i].reset(); 
    numColors = 0;

    for (int v : global_order) {
        if (!cand.test(v)) continue;

        bool placed = false;
        for (int i = 0; i < numColors; i++) {
            if ((colorSets[i] & adj[v]).none()) {
                colorSets[i].set(v);
                colorMax[i] = max(colorMax[i], vertices[v].profit);
                placed = true;
                break;
            }
        }
        if (!placed) {
            colorSets[numColors].reset();
            colorSets[numColors].set(v);
            colorMax[numColors] = vertices[v].profit;
            numColors++;
        }
    }

    for (int i = 0; i < numColors; i++) bound += colorMax[i];
    return bound;
}

int KnapSackBound(const bitset<MAXN> &cand, int remB) {
    if (remB <= 0) return 0;
    double profit = 0;
    int current_w = 0;

    for (int v : ratio_order) {
        if (!cand.test(v)) continue;

        if (current_w + vertices[v].cost <= remB) {
            current_w += vertices[v].cost;
            profit += vertices[v].profit;
        } else {
            double ratio = (double)vertices[v].profit / vertices[v].cost;
            profit += ratio * (remB - current_w);
            break;
        }
    }
    return (int)profit;
}

void FindClique(bitset<MAXN> cand, vi &current_clique, int p_curr, int w_curr) {
    if (p_curr > local_best){
        local_best = p_curr;
        local_best_clique = current_clique;
    }
    if (cand.none()) return;

    if (p_curr + StructuralBound(cand) <= local_best) return;
    if (p_curr + KnapSackBound(cand, B - w_curr) <= local_best) return;

    bitset<MAXN> remaining_cand = cand;
    for (int v : global_order) {
        if (!remaining_cand.test(v)) continue;
        remaining_cand.reset(v); 

        if (w_curr + vertices[v].cost <= B) {
            bitset<MAXN> next_cand = remaining_cand & adj[v];
            current_clique.pb(v);
            FindClique(next_cand, current_clique, p_curr + vertices[v].profit, w_curr + vertices[v].cost);
            current_clique.pop_back();
        }
    }
}

int main(int argc, char** argv) {
    ifstream fin;
    ofstream fout;


    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        if (argc < 3) {
            cerr << "Usage: mpirun -np <p> ./main input.txt output.txt\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fin.open(argv[1]);
        fout.open(argv[2]);

        fin >> N >> E >> B;
    }
    
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&E, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&B, 1, MPI_INT, 0, MPI_COMM_WORLD);

    vertices.resize(N);
    adj.assign(N, bitset<MAXN>());

    if (rank == 0) {
        for (int i = 0; i < N; i++) fin >> vertices[i].profit >> vertices[i].cost;
        for (int i = 0; i < E; i++) {
            int u, v; fin >> u >> v;
            adj[u].set(v); adj[v].set(u);
        }
    }

    MPI_Bcast(vertices.data(), N * sizeof(Vertex), MPI_BYTE, 0, MPI_COMM_WORLD);
    broadcast_graph(rank, size);

    global_order.resize(N); iota(all(global_order), 0);
    sort(all(global_order), [&](int a, int b){ return vertices[a].profit > vertices[b].profit; });
    
    ratio_order.resize(N); iota(all(ratio_order), 0);
    sort(all(ratio_order), [&](int a, int b){
        return (double)vertices[a].profit/vertices[a].cost > (double)vertices[b].profit/vertices[b].cost;
    });

    for (int i = rank; i < N; i += size) {
        int v = global_order[i];
        bitset<MAXN> initial_cand;
        for (int j = i + 1; j < N; j++) {
            if (adj[v].test(global_order[j])) initial_cand.set(global_order[j]);
        }
        vi current_clique = {v};
        FindClique(initial_cand, current_clique, vertices[v].profit, vertices[v].cost);
    }
    struct { int val; int rank; } local_res, global_res;
    local_res.val = local_best;
    local_res.rank = rank;

    MPI_Allreduce(&local_res, &global_res, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

    if (rank == 0) {
        if (global_res.rank == 0) {
            fout << local_best << "\n";
            sort(all(local_best_clique));
            for(int v : local_best_clique) fout << v << " ";
            fout << endl;
        } else {
            int clique_size;
            MPI_Recv(&clique_size, 1, MPI_INT, global_res.rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            vi final_clique(clique_size);
            MPI_Recv(final_clique.data(), clique_size, MPI_INT, global_res.rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            fout << global_res.val << "\n";
            sort(all(final_clique));
            for(int v : final_clique) fout << v << " ";
            fout << endl;
        }
    } else if (rank == global_res.rank) {
        int clique_size = local_best_clique.size();
        MPI_Send(&clique_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(local_best_clique.data(), clique_size, MPI_INT, 0, 1, MPI_COMM_WORLD);
    }
    if (rank == 0) {
        fin.close();
        fout.close();
    }

    MPI_Finalize();
    return 0;
}