#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
#include <limits>
#include <set>
#include <numeric>

class Graph {
private:
    int V;
    std::vector<std::vector<std::pair<int, int>>> adj;  // {vertex, weight}
    std::vector<std::vector<int>> unweighted_adj;

public:
    Graph(int v) : V(v), adj(v), unweighted_adj(v) {}

    void addEdge(int u, int v, int w) {
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
        unweighted_adj[u].push_back(v);
        unweighted_adj[v].push_back(u);
    }

    void printGraph() const {
        for (int i = 0; i < V; ++i) {
            std::cout << i << ": ";
            for (const auto& [v, w] : adj[i]) {
                std::cout << "(" << v << ", " << w << ") ";
            }
            std::cout << std::endl;
        }
    }

    bool isEulerGraph() const {
        for (const auto& neighbors : unweighted_adj) {
            if (neighbors.size() % 2 != 0) {
                return false;
            }
        }
        return true;
    }

    std::vector<int> fleuryAlgorithm() {
        if (!isEulerGraph()) {
            return {};
        }

        std::vector<std::vector<int>> adj_copy = unweighted_adj;
        std::vector<int> circuit;
        int start = 0;

        auto dfs = [&](auto&& self, int u) -> void {
            for (int v : adj_copy[u]) {
                if (v != -1) {
                    adj_copy[u].erase(std::remove(adj_copy[u].begin(), adj_copy[u].end(), v), adj_copy[u].end());
                    adj_copy[v].erase(std::remove(adj_copy[v].begin(), adj_copy[v].end(), u), adj_copy[v].end());
                    self(self, v);
                }
            }
            circuit.push_back(u);
        };

        dfs(dfs, start);
        std::reverse(circuit.begin(), circuit.end());
        return circuit;
    }

    std::vector<int> dijkstra(int start) {
        std::vector<int> dist(V, std::numeric_limits<int>::max());
        std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> pq;

        dist[start] = 0;
        pq.push({0, start});

        while (!pq.empty()) {
            int u = pq.top().second;
            pq.pop();

            for (const auto& [v, w] : adj[u]) {
                if (dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    pq.push({dist[v], v});
                }
            }
        }

        return dist;
    }

    std::vector<std::pair<int, int>> primMST() {
        std::vector<bool> inMST(V, false);
        std::vector<int> key(V, std::numeric_limits<int>::max());
        std::vector<int> parent(V, -1);
        std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> pq;

        key[0] = 0;
        pq.push({0, 0});

        while (!pq.empty()) {
            int u = pq.top().second;
            pq.pop();

            inMST[u] = true;

            for (const auto& [v, w] : adj[u]) {
                if (!inMST[v] && w < key[v]) {
                    parent[v] = u;
                    key[v] = w;
                    pq.push({key[v], v});
                }
            }
        }

        std::vector<std::pair<int, int>> mst;
        for (int i = 1; i < V; ++i) {
            mst.push_back({parent[i], i});
        }
        return mst;
    }

    
    std::vector<std::vector<int>> fundamentalCutsets(const std::vector<std::pair<int, int>>& mst) {
        std::vector<std::vector<int>> cutsets;

        for (const auto& [u, v] : mst) {
            // Temporarily remove edge u-v
            // Perform DFS/BFS to find reachable nodes from u
            // The nodes not reachable from u are part of the cutset
            std::vector<bool> visited(V, false);
            
            auto dfs = [&](auto&& self, int x) -> void {
                visited[x] = true;
                for (const auto& [y, w] : adj[x]) {
                    if (!visited[y] && !(x == u && y == v) && !(x == v && y == u)) {
                        self(self, y);
                    }
                }
            };

            dfs(dfs, u);

            // Collect nodes not visited from u
            std::vector<int> cutset;
            for (int i = 0; i < V; ++i) {
                if (!visited[i]) cutset.push_back(i);
            }
            
            cutsets.push_back(cutset);
        }
        
        return cutsets;
    }

   
    std::vector<std::vector<int>> fundamentalCircuits(const std::vector<std::pair<int, int>>& mst) {
        std::vector<std::vector<int>> circuits;

        // Iterate through each edge in the MST
        for (const auto& [u, v] : mst) {
            // Create a path from u to v using DFS
            // Add the edge u-v back to the graph temporarily
            addEdge(u,v,std::numeric_limits<int>::max()); // Temporarily add edge

            // Find a circuit using DFS starting from u
            std::set<int> visited; 
            
            auto dfs = [&](auto&& self, int x) -> void { 
                visited.insert(x); 
                for(const auto& [y,w]:adj[x]){ 
                    if(visited.find(y)==visited.end()){ 
                        self(self,y); 
                    } 
                } 
                circuits.back().push_back(x); 
             };

             circuits.push_back(std :: vector<int>{}); // Start new circuit with node u
             dfs(dfs,u); // Start DFS from u

             removeEdge(u,v); // Remove edge after finding circuit
         }
         return circuits; 
     }

     void removeEdge(int u,int v){ 
         for(auto it=adj[u].begin(); it!=adj[u].end();){ 
             if(it->first==v){ 
                 it=adj[u].erase(it); 
             }else{ 
                 ++it; 
             } 
         } 

         for(auto it=adj[v].begin(); it!=adj[v].end();){ 
             if(it->first==u){ 
                 it=adj[v].erase(it); 
             }else{ 
                 ++it; 
             } 
         } 
     }

     int getV() const { return V; }
};

Graph havelHakimi(std::vector<int>& sequence) {
    int n = sequence.size();
    std::vector<std::pair<int, int>> degrees;

    for (int i = 0; i < n; ++i) {
        if (sequence[i] >= n) {
            std::cout << "Degree " << sequence[i] << " at position " << i << " is too large for a graph with " << n << " vertices." << std::endl;
            return Graph(0);
        }
        degrees.push_back({sequence[i], i});
    }

    Graph G(n);

    while (!degrees.empty()) {
        std::sort(degrees.begin(), degrees.end(), std::greater<>());
        
        int d = degrees[0].first;
        int v = degrees[0].second;
        
        degrees.erase(degrees.begin());

       if (d > degrees.size()) { 
           std :: cout<<"Not enough vertices to connect to vertex "<<v<<" with degree "<<d<<"."<<std :: endl ; 
           return Graph(0);  
       } 

       static std :: vector<std :: set<int>> connections(n); 

       for (int i = 0; i < d; ++i) { 
           degrees[i].first--; 

           if(degrees[i].first<0){ 
               std :: cout<<"Negative degree encountered for vertex "<<degrees[i].second<<"."<<std :: endl ; 
               return Graph(0);  
           } 

           int u=degrees[i].second; 

           if(connections[v].find(u)==connections[v].end()){ 
               connections[v].insert(u); 
               connections[u].insert(v); 

               int weight=rand() % 100 + 1;  // Random weight between 1 and 100
               G.addEdge(v,u,weight);  
           }  
       }  
   }  

   return G;  
}  

int main() {  
   srand(static_cast<unsigned>(time(0)));  

   // Ask user to input graphical degree sequence  
   int n;  
   std :: cout<<"Enter the number of vertices: ";  
   std :: cin>>n;  

   std :: vector<int> sequence(n);  
   std :: cout<<"Enter the degree sequence: ";  
   for(int i=0;i<n;++i){  
       std :: cin>>sequence[i];  
   }  

   Graph G=havelHakimi(sequence);  
   if(G.getV()==0){  
       std :: cout<<"The sequence is not graphical."<<std :: endl;  
       return 0;  
   }  

   std :: cout<<"Graph created successfully."<<std :: endl;  
   G.printGraph();  

   // Test if the graph is Eulerian   
   if(G.isEulerGraph()){   
       std :: cout<<"The graph is Eulerian."<<std :: endl;   
       auto eulerCircuit=G.fleuryAlgorithm();   
       std :: cout<<"Euler circuit: ";   
       for(int v:eulerCircuit){   
           std :: cout<<v<<" ";   
       }   
       std :: cout<<std :: endl;   
   } else {   
       std :: cout<<"The graph is not Eulerian."<<std :: endl;   
   }   

   // Dijkstra's algorithm    
   int start=rand()%n;    
   std :: cout<<"Single source shortest paths from vertex "<<start<<":"<<std :: endl;    
   auto shortestPaths=G.dijkstra(start);    

   for(int i=0;i<n;++i){    
       if(shortestPaths[i]==std :: numeric_limits<int>::max()){    
           shortestPaths[i]=-1;// Indicate unreachable nodes    
       }    
       std :: cout <<"Distance to "<<i<<": "<<shortestPaths[i]<<std :: endl ;    
     }    

     // Minimum Spanning Tree    
     auto mst=G.primMST();    
     if(mst.empty()){    
         std :: cout<<"No MST exists"<<std :: endl;    
     }else{    
         std :: cout<<"Minimum Spanning Tree edges:"<<std :: endl;    
         for(const auto &[u,v]:mst){    
             std :: cout<<u<<" - "<<v<<std :: endl ;    
         }  
     }    

}
#include <set>
#include <functional>

using namespace std;


struct Edge {
    int src, dest, weight;
};


int findParent(int i, vector<int>& parent) {
    if (parent[i] == -1)
        return i;
    return findParent(parent[i], parent);
}


void unionSets(int x, int y, vector<int>& parent) {
    parent[x] = y;
}


void primMST(vector<vector<int>>& graph) {
    int V = graph.size();
    vector<int> parent(V, -1);
    vector<bool> inMST(V, false);
    vector<int> key(V, INT_MAX);
    
    key[0] = 0;

    for (int count = 0; count < V - 1; count++) {
        int minKey = INT_MAX, minIndex;
        for (int v = 0; v < V; v++)
            if (!inMST[v] && key[v] < minKey) {
                minKey = key[v];
                minIndex = v;
            }

        inMST[minIndex] = true;

        for (int v = 0; v < V; v++)
            if (graph[minIndex][v] && !inMST[v] && graph[minIndex][v] < key[v]) {
                parent[v] = minIndex;
                key[v] = graph[minIndex][v];
            }
    }

    
    cout << "Prim's Algorithm MST:\n";
    cout << "Edge \tWeight\n";
    int totalWeightPrim = 0;
    for (int i = 1; i < V; i++) {
        cout << parent[i] << " - " << i << "\t" << graph[i][parent[i]] << endl;
        totalWeightPrim += graph[i][parent[i]];
    }
    cout << "Total Weight: " << totalWeightPrim << endl;

    
    cout << "\nFundamental Cutsets:\n";
    for (int i = 1; i < V; i++) {
        cout << "Cutset for edge " << parent[i] << " - " << i << ": { ";
        bool first = true; // Flag to handle comma placement
        for (int j = 0; j < V; j++) {
            if (graph[parent[i]][j] > 0 && j != i) {
                if (!first) {
                    cout << ", "; // Add comma before subsequent edges
                }
                cout << parent[i] << " - " << j;
                first = false; // After the first edge, set flag to false
            }
        }
        cout << " }\n";
    }
}

// Kruskal's Algorithm to find MST
void kruskalMST(vector<Edge>& edges, int V) {
    sort(edges.begin(), edges.end(), [](Edge a, Edge b) { return a.weight < b.weight; });
    
    vector<int> parent(V, -1);
    
    cout << "\nKruskal's Algorithm MST:\n";
    cout << "Edge \tWeight\n";
    
    int totalWeightKruskal = 0;
    
    for (const auto& edge : edges) {
        int x = findParent(edge.src, parent);
        int y = findParent(edge.dest, parent);

        if (x != y) {
            cout << edge.src << " - " << edge.dest << "\t" << edge.weight << endl;
            unionSets(x, y, parent);
            totalWeightKruskal += edge.weight;
        }
    }
    
    cout << "Total Weight: " << totalWeightKruskal << endl;

   
    cout << "\nFundamental Circuits:\n";
    
    for (const auto& edge : edges) {
        int x = findParent(edge.src, parent);
        int y = findParent(edge.dest, parent);

      
        if (x == y) {
            cout << "Circuit formed by adding edge: " << edge.src << " - " << edge.dest << endl;

          
            vector<int> circuitVertices;
            set<int> visited;

           
            function<void(int)> dfs = [&](int v) {
                visited.insert(v);
                circuitVertices.push_back(v);
                for (const auto& e : edges) {
                    if ((e.src == v || e.dest == v) && visited.find(e.dest) == visited.end() && visited.find(e.src) == visited.end()) {
                        dfs(e.dest);
                    }
                }
            };

            dfs(edge.src); // Start DFS from one end of the added edge
            
            cout << "{ ";
            for (size_t i = 0; i < circuitVertices.size(); ++i) {
                cout << circuitVertices[i];
                if (i < circuitVertices.size() - 1) { // Add comma if not last element
                    cout << ", ";
                }
            }
            cout << ", " << edge.src << ", " << edge.dest; // Include the added edge vertices
            cout << " }\n";
        }
    }
}

int main() {
   
    vector<vector<int>> graph = {
        {0, 2, 0, 6, 0},
        {2, 0, 3, 8, 5},
        {0, 3, 0, 0, 7},
        {6, 8, 0, 0, 9},
        {0, 5, 7, 9, 0}
    };
    
   
    vector<Edge> edges = {
        {0, 1 ,2}, {1 ,2 ,3}, {0 ,3 ,6}, 
        {1 ,3 ,8}, {1 ,4 ,5}, {2 ,4 ,7}, 
        {3 ,4 ,9}
    };
    
    primMST(graph);
    
    int V = graph.size(); // Number of vertices
    kruskalMST(edges,V);

    return 0;
}
#include <iostream>
#include <vector>
#include <limits.h>
#include <queue>
#include <algorithm>

using namespace std;

class Graph {
public:
    int V;
    vector<vector<int>> adj; 

    Graph(int V);
    void addEdge(int u, int v);
    int vertexConnectivity();
    int edgeConnectivity();
    bool bfs(const vector<vector<int>>& rGraph, int s, int t, vector<int>& parent);
    int fordFulkerson(vector<vector<int>>& graph, int s, int t);
    void printGraph(); 
};


Graph::Graph(int V) {
    this->V = V;
    adj.resize(V);
}

void Graph::addEdge(int u, int v) {
    adj[u].push_back(v);
    adj[v].push_back(u); // For undirected graph
}
bool Graph::bfs(const vector<vector<int>>& rGraph, int s, int t, vector<int>& parent) {
    int n = rGraph.size();
    vector<bool> visited(n, false);
    queue<int> q;
    q.push(s);
    visited[s] = true;
    parent[s] = -1;

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int v = 0; v < n; v++) {
            if (!visited[v] && rGraph[u][v] > 0) {
                if (v == t) {
                    parent[v] = u;
                    return true;
                }
                q.push(v);
                parent[v] = u;
                visited[v] = true;
            }
        }
    }
    return false;
}
int Graph::fordFulkerson(vector<vector<int>>& graph, int s, int t) {
    int n = graph.size();
    vector<vector<int>> rGraph = graph;
    vector<int> parent(n);
    int max_flow = 0;

    while (bfs(rGraph, s, t, parent)) {
        int path_flow = INT_MAX;
        for (int v = t; v != s; v = parent[v]) {
            int u = parent[v];
            path_flow = min(path_flow, rGraph[u][v]);
        }

        for (int v = t; v != s; v = parent[v]) {
            int u = parent[v];
            rGraph[u][v] -= path_flow;
            rGraph[v][u] += path_flow;
        }

        max_flow += path_flow;
    }

    return max_flow;
}

int Graph::vertexConnectivity() {
    if (V <= 1) return 0;
    if (V == 2) return adj[0].size() > 0 ? 1 : 0;

    int min_cut = INT_MAX;
    for (int s = 0; s < V - 1; s++) {
        for (int t = s + 1; t < V; t++) {
            vector<vector<int>> graph(V + 2, vector<int>(V + 2, 0));
            int source = V, sink = V + 1;

            for (int i = 0; i < V; i++) {
                if (i != s && i != t) {
                    graph[source][i] = 1;
                    graph[i][sink] = 1;
                }
                for (int j : adj[i]) {
                    if (i != s && i != t && j != s && j != t) {
                        graph[i][j] = 1;
                    }
                }
            }

            int cut = fordFulkerson(graph, source, sink);
            min_cut = min(min_cut, cut);
        }
    }
    return min_cut;
}

int Graph::edgeConnectivity() {
    vector<vector<int>> graph(V, vector<int>(V, 0));
    for (int u = 0; u < V; u++) {
        for (int v : adj[u]) {
            graph[u][v] = 1;
        }
    }

    int min_cut = INT_MAX;
    for (int s = 0; s < V - 1; s++) {
        for (int t = s + 1; t < V; t++) {
            min_cut = min(min_cut, fordFulkerson(graph, s, t));
        }
    }
    return min_cut;
}


void Graph::printGraph() {
    for (int i = 0; i < V; i++) {
        cout << "Vertex " << i << ":";
        for (int neighbor : adj[i]) {
            cout << " " << neighbor;
        }
        cout << endl;
    }
}

int main() {
    Graph g(4); 
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(0, 3);
    g.addEdge(1, 2);
    g.addEdge(1, 3);
    g.addEdge(2, 3);

    cout << "Graph structure:" << endl;
    g.printGraph();

    int vertexConn = g.vertexConnectivity();
    cout << "Vertex Connectivity: " << vertexConn << endl;

    int edgeConn = g.edgeConnectivity();
    cout << "Edge Connectivity: " << edgeConn << endl;

    int k = min(vertexConn, edgeConn);
    cout << "The graph is " << k << "-connected." << endl;

    return 0;
}