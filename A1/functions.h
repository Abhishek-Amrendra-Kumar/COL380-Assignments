#pragma once

#include <iostream>
#include <string>
#include <cstdint>
#include <vector>
#include <iomanip>
#include <fstream>
#include <map>
#include <algorithm>
#include <cmath>

void updateDisplay(const std::vector<uint64_t> &orderBook, int32_t freq);
int64_t totalAmountTraded(const std::vector<uint64_t> &orderBook);
void printOrderStats(const std::vector<uint64_t> &orderBook);