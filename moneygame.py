import random
    
def exchange(people, i, j, ratio):
    sumM = people[i]+people[j]
    people[i] = sumM*ratio
    people[j] = sumM*(1-ratio)
    
    
def main():
    people = []
    ratio = .2
    trials = 100
    initial = 1000
    
    for x in range(100):
        people[x] = initial
        
    print(people)
    
    for x in range(trials):
        i = random(people.length)
        j = random(people.length)
    
        #if i != j:
        exchange(people,i,j,ratio)
       
    print(people)
    