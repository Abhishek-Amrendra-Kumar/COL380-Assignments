#include "functions.h"
#include<vector>
#include<algorithm>
#include<fstream>
#include<omp.h>
#include<cstdint>
#include<iomanip>
#include<cmath>
uint64_t unstuffBits(uint64_t packet) {
    uint64_t result = 0;
    int outputBitPos = 0;
    int consecutiveOnes = 0;
    const int TARGET_BITS = 49;
    
    int i = 0;
    while (i < 64 && outputBitPos < TARGET_BITS) {
        bool bit = (packet >> i) & 0x1;
        
        if (bit) {
            consecutiveOnes++;
            result |= (1ULL << outputBitPos);
            outputBitPos++;
            
            if (consecutiveOnes == 5) {
                consecutiveOnes = 0;
                i+=2; 
                continue;
            }
        } else {
            consecutiveOnes = 0;
            outputBitPos++;
        }
        i++;
    }
    
    return result;
}

struct Decoded{
    uint32_t stock_id;
    bool order_type;
    uint8_t order_value; 
    uint8_t order_quantity;
};

static inline __attribute__((always_inline))
Decoded decode_packet(uint64_t packet) {
    uint32_t stock_id;
    bool order_type;
    uint8_t order_value;
    uint8_t order_quantity;
    stock_id = (packet >> 0) & 0xFFFFFFFF;
    order_type = ((packet >> 32) & 0x1) == 1;
    order_value = (packet >> 41) & 0xFF;
    order_quantity = (packet >> 33) & 0xFF;
    return {stock_id,order_type,order_value,order_quantity};
}

const int MAX_ID = 1'000'001;

struct stock_info{
    uint32_t stock_id;
    uint8_t last_sell;
    uint8_t last_buy;
    uint8_t spread;
};

static void take_snapshot(
    int snap_no,
    const std::vector<uint8_t> &last_buy,
    const std::vector<uint8_t> &last_sell,
    const std::vector<uint32_t> &active_stocks){
        std::vector<stock_info> stocks;
        stocks.reserve(active_stocks.size());
        for(uint32_t id: active_stocks){
             uint8_t spread = static_cast<uint8_t>(
            std::abs((int)last_buy[id] - (int)last_sell[id])
        );
            stocks.push_back({id,last_sell[id], last_buy[id], spread});
        }
        std::sort(stocks.begin(),stocks.end(), 
            [](const stock_info& stock1, const stock_info& stock2){
                if(stock1.spread != stock2.spread) return stock1.spread > stock2.spread;
                return stock1.stock_id > stock2.stock_id; 
            });
        std::ofstream out("snap_" + std::to_string(snap_no)+".txt");
        for(auto&s : stocks){
            out << s.stock_id << " "
            << (int)s.last_sell << " "
            << (int)s.last_buy << " "
            << (int)s.spread << "\n";
        }
    };


void updateDisplay(const std::vector<uint64_t> &orderBook, int32_t freq){
    uint64_t size = orderBook.size();
    
    std::vector<Decoded> decoded(size);

    #pragma omp parallel for schedule(static)
        for(uint64_t i = 0; i < size;i++){
            uint64_t unstuffed = unstuffBits(orderBook[i]);
            decoded[i] = decode_packet(unstuffed);
        }
    std::vector<uint8_t> last_buy(MAX_ID,0);
    std::vector<uint8_t> last_sell(MAX_ID,0);
    std::vector<uint8_t> active(MAX_ID,0);
    
    std::vector<uint32_t> active_stocks;
    active_stocks.reserve(100000);
    
    int curr = 0;
    #pragma omp parallel
    {
        #pragma omp single
        {
            for(uint64_t i = 0; i < size; i++){
                    auto& pkt = decoded[i];
                    if(!active[pkt.stock_id]){
                        active[pkt.stock_id] = 1;
                        active_stocks.push_back(pkt.stock_id);
                    }
                    if(pkt.order_type){
                        last_sell[pkt.stock_id] = pkt.order_value;
                    }else{
                        last_buy[pkt.stock_id] = pkt.order_value;
                    }

                    if(((i+1) %freq == 0)|| (i == size-1)){
                        std::vector<uint8_t> snapshot_buy = last_buy;
                        std::vector<uint8_t> snapshot_sell = last_sell;
                        std::vector<uint32_t> snap_active = active_stocks;

                        int snap_no = curr++;
                        #pragma omp task firstprivate(snap_no, snapshot_buy, snapshot_sell, snap_active)
                        {
                            take_snapshot(snap_no,snapshot_buy,snapshot_sell,snap_active);
                        }

                    }
            }
            #pragma omp taskwait 
        } 
    } 
}
int64_t totalAmountTraded(const std::vector<uint64_t> &orderBook)
{
    uint64_t size = orderBook.size();
    int64_t total = 0;
    #pragma omp parallel for reduction(+:total) schedule(static)
        for(uint64_t i = 0; i < size; i++){
            uint64_t unstuffed = unstuffBits(orderBook[i]);
            Decoded pkt = decode_packet(unstuffed);
            total += (int64_t) pkt.order_value* (int64_t)pkt.order_quantity;
        }
    return total;
}

struct stock_info2{
    uint8_t min_sell;
    uint8_t max_buy;
    bool has_sell;
    uint64_t sum;
    uint64_t count;
    stock_info2(): min_sell(255),max_buy(0),sum(0),count(0),has_sell(0){}
};
void printOrderStats(const std::vector<uint64_t> &orderBook)
{
    uint64_t size = orderBook.size();
    
    int num_threads = omp_get_max_threads();
    std::vector<std::vector<stock_info2>> local(num_threads,std::vector<stock_info2>(MAX_ID));

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        auto &current = local[thread_id];
        #pragma omp for schedule(static)
            for(uint64_t i = 0; i < size; i++){
                
                uint64_t unstuffed = unstuffBits(orderBook[i]);
                Decoded pkt = decode_packet(unstuffed);
                stock_info2& s = current[pkt.stock_id];

                if(pkt.order_type){
                    s.has_sell = 1;
                    s.min_sell = std::min(s.min_sell, pkt.order_value);
                }else{
                    s.max_buy = std::max(s.max_buy, pkt.order_value);
                }
                s.sum += pkt.order_value;
                s.count++;
            }
    }
    std::vector<stock_info2> global(MAX_ID);

    #pragma omp parallel for schedule(static)
    for(int id = 0; id < MAX_ID; id++){
        stock_info2 g;
        for(int t = 0; t < num_threads; t++){
            g.max_buy = std::max(g.max_buy,local[t][id].max_buy);
            if(local[t][id].has_sell){
                g.min_sell = std::min(g.min_sell, local[t][id].min_sell);
                g.has_sell = true;
            }
            g.sum += local[t][id].sum;
            g.count += local[t][id].count;
        }
        global[id] = g;
    }
    std::ofstream out("stats.txt");
    for(int id = 0; id < MAX_ID; id++){
        if(global[id].count > 0){
            double avg = (double)global[id].sum / global[id].count;
            if(!global[id].has_sell) global[id].min_sell = 0;
            out << id << " " << (int)global[id].min_sell<<" " << (int)global[id].max_buy << " "
            << std::fixed << std::setprecision(3) << avg << "\n";   
        }
    }
    
}