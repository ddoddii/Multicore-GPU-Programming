#include <iostream>
#include <vector>
using namespace std;

void koggeStoneScan(vector<int> &input)
{
    int n = input.size();
    vector<int> output(n, 0);
    output = input;

    for (int d = 1; d < n; d *= 2)
    {
        vector<int> temp(output);

        for (int i = d; i < n; ++i)
        {
            output[i] = temp[i] + temp[i - d];
        }
    }
    input = output;
}

int main()
{
    vector<int> data = {3, 1, 7, 0, 4, 1, 6, 3};
    cout << "Original data : ";
    for (int num : data)
    {
        cout << num << " ";
    }

    koggeStoneScan(data);

    cout << "\nPrefix sum (Kogge-stone) : ";
    for (int num : data)
    {
        cout << num << " ";
    }

    cout << endl;
    return 0;
}