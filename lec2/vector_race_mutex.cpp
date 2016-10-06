#include <iostream>
#include <vector>
#include <thread>
#include <mutex>

using namespace std;

vector<int> v;
mutex m;

void addElements() 
{
    m.lock();
    v.push_back(1);
    m.unlock();
}
  
int main ()
{
    thread threads[100];
    
    for(int i = 0; i < 100; i++)
        threads[i] = thread(addElements);
    
    for(int i = 0; i < 100; i++)
        threads[i].join();
    
    cout << v.size() << endl;
}
