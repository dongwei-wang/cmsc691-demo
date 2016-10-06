#include <iostream>
#include <vector>
#include <thread>

using namespace std;

vector<int> v;

void addElements() 
{
    v.push_back(1);
}
  
int main ()
{
    thread threads[4];
    
    for(int i = 0; i < 4; i++)
        threads[i] = thread(addElements);
    
    for(int i = 0; i < 4; i++)
        threads[i].join();
    
    cout << v.size() << endl;
}
