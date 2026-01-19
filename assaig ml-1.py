#machine learning assaignment-1 (odd)

import random
#Q1
#Counting pairs whose sum is 10
def count_pairs_with_sum(numbers):
    pair_count = 0

    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if numbers[i] + numbers[j] == 10:
                pair_count = pair_count + 1

    return pair_count
#Q2
#Range of real numbers
def find_range(values):
    if len(values)<3:
        return "range determination not possible"
    minimum=values[0]
    maximum=values[0]

    for num in values:
        if num<minimum:
            minimum=num
        if num>maximum:
            maximum=num

    return maximum-minimum
#Q3
#matrixpower A^m
def multiply_matrices(a,b):
    size=len(a)
    result=[]

    for i in range(size):
        row=[]
        for j in range(size):
            t=0
            for k in range(size):
                t=t+a[i][k]*b[k][j]
            row.append(total)
        result.append(row)
        
    return result

def matrix_power(a,m):
    size=len(a)
    
#identity matrix
    result=[]
    for i in range(size):
        row=[]
    for i in range(size):
        if i==j:
            row.append(1)
        else:
            row.append(0)
        result.append(row)
    for _ in range (m):
        result=multiply_matrices(result,a)

    return result

#Q4
#highly occuring letter char
    def higest_occuringchar(text):
        freq={}
        for ch in text:
            if ch.isalpha():
                freq[ch]=freq[ch]+1
            else:
                freq[ch]=1
    max_char=None
    max_count=0
    for ch in freq:
        if freq[ch]>max_count:
            max_count=freq[ch]
            max_char=ch

            return max_char,max_count

#Q5
#mean,median,mode of 25randomnumbers
def calculate_statistics():
    numbers = []

    for i in range(25):
        numbers.append(random.randint(1, 10))

    numbers.sort()

    # Mean
    total = 0
    for num in numbers:
        total = total + num
    mean = total / len(numbers)

    # Median
    median = numbers[len(numbers) // 2]

    # Mode
    frequency = {}
    for num in numbers:
        if num in frequency:
            frequency[num] = frequency[num] + 1
        else:
            frequency[num] = 1

    mode = numbers[0]
    max_count = 0

    for num in frequency:
        if frequency[num] > max_count:
            max_count = frequency[num]
            mode = num

    return numbers, mean, median, mode



#main functions
def main():
   # Question 1
    numbers_q1 = [2, 7, 4, 1, 3, 6]
    print("Q1: Number of pairs with sum 10:",
          count_pairs_with_sum(numbers_q1))


    # Question 2
    values_q2 = [5, 3.8, 10, 4]
    print("Q2: Range of the list:",
          find_range(values_q2))

    # Question 3
    matrix = [
        [1, 2],
        [3, 4]
    ]
    power = 2
    print("Q3: Matrix power result:")
    result_matrix = matrix_power(matrix, power)
    for row in result_matrix:
        print(row)

    # Question 4
    text = "hippopotamus"
    char, count = highest_occurring_character(text)
    print("Q4:", char, "occurs", count, "times")

    # Question 5
    nums, mean, median, mode = calculate_statistics()
    print("Q5: Generated numbers:", nums)
    print("Mean:", mean)
    print("Median:", median)
    print("Mode:", mode)


main()

