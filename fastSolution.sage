'''

Remmeber the Goal is to calculate using the Barycentric Form which would be super fast


This is the basic setup for the problem.
We work on the finite Field of P. In our simple case all we need to understand is that a finite
field lets you to +-/* under the modulus of P. This essentially make it so no number can be larger
then our P

This P is given to us from the challenge, all this is the basic setup

R gives us polynomials with variable X with coefficientts mod P
This way later when we want to multiply two polynomials it is
fastly optimized and mod P
'''

p = 7514777789
X = []
Y = []

for line in open('encoded.txt', 'r').read().strip().split('\n'):
    x, y = line.split(' ')
    X.append(int(x))
    Y.append(int(y))
K = GF(p)
R = PolynomialRing(K, 'x')

'''
Now we create the Class Tree for Z(X) heres the binary tree implementation
Remember that Z(X) is the poly representation of (x - Xi) for all X points multiplied
'''

class Tree():
    def __init__(self, poly, X, left=None, right=None):
        self.left = left # Left Child
        self.right = right #Right Child
        self.poly = poly # Polynomial for thise node
        self.X = X # Which x-values this polynomial correspond to

    #Lets us find the lenght very easily by doing len(tree)
    #We know we are a lead node if the length == 1 because theirs only 1 Xi

    def __len__(self):
        return len(self.X)
    
    
    '''
    This multiplication function does four different things at once
    1. Multiplies the polynomials very fast use Sages Internal Polynomial multiplication (uses FFT)
    2. merges the lists of X so we know all the X inside
    3. creates a parent node to self and other
    4. Returns the new parent node

    SO now we have that a parent equal to its left * right children

    '''
    
    def __mul__(self, other):
        return Tree(
            self.poly * other.poly,
            self.X + other.X,
            self,
            other
        )
    
'''

Now we are going to compute the tree Z(X)
Remember here that R is just our polynomial variable x that is bounded under that modulus.
'''

def compTree(X):
    x = R.gen() #Grabs the polynomial x from Sage

    nodes = []
    for xK in X:
        nodes.append(Tree(R(x - xK), [xK])) #x - xK is literally how sage makes polynomials
        #We have the symbol x and we subtract the points xK for all the X's we have
        #Then putting it in the R function lets us easily enforce the modulus rule

    #Loop through all the points and start multiplying them to make their parents
    #We keep combining until only two left, then we return the root
    #Its a very simple construction
    while (len(nodes) > 2):
        new_nodes = []
        for j in range(0, len(nodes)-1, 2):
            node = nodes[j] * nodes[j+1]
            new_nodes.append(node)
        if len(nodes) % 2 == 1:
            new_nodes.append(nodes[-1]) #If we had Odd nodes then we need to just throw in that left over leaf for next computation
        nodes = new_nodes

    return nodes[0] * nodes[1] # This is the root


'''
Now lets say we want to evalute one of these polynomials from our tree
We need a fast way to do it insted of plugging each point in

Theirs a math fact gained from a book called modern computer algebra and his writeup which claims

f(xi) = remainder of f(x) when divided by (x - xi)

simple explanation

f(x) = q(x)(x - xi) + r

=> f(xi) = r

for example lets say you have f(x) = x^2 + 5x + 4

If you want to evalute f(2) the slow way can be found like so 4 + 10 + 4 = 18
But you can do a simple math trick. Rearrange the equation to be (x+7)(x-2) + 18.
Now if you want to calculate f(2) its simply shows to be 18 as the polynomial part becomes 0.
So you can always wight F(X) = Q(X)(X - Xi) + R
If we Divide F(X) by (X-Xi) its remainder is simply R witch is F(Xi)
Theirfore F(Xi) = Rem(F(X)/(X-Xi))
We will know use our tree we made to easily calculate this division and get remainders/evals fast!

This fast Eval is going to try and compute all f(xi) from our tree strucutre
so if one of the leaf nodes is a child of that parent f then it will eventually compute and return

Example F(X) = x^3 + 5x + 1

Z(X) = (x - 1)(x - 2)(x - 3)(x - 7)

At the root level we split th epoints into two groups
[1, 2] and [3, 7]
So we divide F(X) by (x - 1)(x - 2) with remainder r1(x)
and (x-3)(x-7) with remainder r2(x)

Why?

if x = 1 or x = 2 then (x-1)(x-2) = 0 so for those points the remainder
of the function would be the evaluation at that x

Remember these are from our Tree We constructed!

A question arrises why dont we just do
for each x_i:
    compute remainder of f(x) /(mod) (x - x_i) ??


    this arrises at the original problem because if f(x) is degree N, we have to do N divisions for each coefficient. 
    And we have to do this N times so it becomes (On^2) this is not good

    For the tree, we divide by a huge polynomial early on, and compute its remainder

    When we divide by half of the coefficients we use this polynomial theorem
    If we assume F(X) = Q(X)RL(X) + R(X) and F(X) = Q(X)RR(X) + R(X) which is allowed
    then assume pluggin in for your inputted amount on the leftside and rightside you get
    F(Xr) = R(Xr) and same thing for the left side.

    We just continue this computation until the remainder itself is degree 0
    Or the tree becomes the leadNode in which in that case 

WOrked Example:

f(x) = x^2 + 1
X = [1, 3]

We want to eval F(X) for all our X points

first build the tree

   (x-1)(x - 3)

(x-1)      (x-3)

Notice that the root is not the base case. The Root Poly degree is = 2

Now we divide f(x) by left and right

x^2 + 1 = (x + 1)(x - 1) + 2 theirfore remainder 2

r1(x) = 2

x² + 1 = (x + 3)(x - 3) + 10

r2(x) = 10

it calls itself with both these recusirve cases,
Base case hits because tree polynomial is degree 1,

We just evaluate the new F(the X we want) where that X is from the lad Node
We return an array of that answer of F(that X) and add em all up

Again remember that F(X) = Q(X)G(X) + R(X)
if we wanted F(A) or F(B) and G(A) and G(B) = 0 then
we know that F(A) or F(B) will equal R(X=A or B).
When we reacha tree node of polynomial degree 1
Well we should just evalute for that X point at this point with our simplfiied R(X)
Sometimes though we will get a case where the Remainder is a Number, that just means
No matter the input of R(A or B) it will always gives the same answer.
This makes sense like think of 0's of a function.

The order of the recusion also gurantees that the F(Xis) are returned in the 
correct order
'''

def fastEval(f, tree):
    if f.degree() < 2 or tree.poly.degree() < 2:
        if tree.poly.degree() == 1:
            return [f(-tree.poly(0))] # Case where we use our simplified poly R(X) for a given Xi
        if f.degree() == 0:
            return [f] #Case where multiple Xis may have the same output! (Like 0s)
        
    A = B = 0

    '''
    At a node representing Z(x) = ZL(x) * ZR(X)
    Divide F(X) by the leftside to get R1
    Divide F(X) by the rightside to get R2
    
    '''
    if tree.left:
        _, r1 = f.quo_rem(tree.left.poly)
        A = fastEval(r1, tree.left)

    if tree.right:
        _, r2 = f.quo_rem(tree.right.poly)
        B = fastEval(r2, tree.right)

    return A + B

'''
Calc Weights

We know that the Mult of every j for (xi - xj)
is equal to the deriviate of Z'(xi)
This is prooven for the barycentric version of Z
In the equation the weights become

The weights become wi = yi/Z'(xi)

'''

def calcWeights(X, Y, tree):
    Zp = tree.poly.derivative()
    denom = fastEval(Zp, tree)
    return [y / d for y, d in zip(Y, denom)]

'''
Remember our P(X) is simply
the summation of Wi Z(X) / x - xi

We already found a way to calculate all of Wi Really quickly
We know just do a divide and conquer teqniues for all Xi using the subproduct tree

Again to reiterate -> Look up barycentric form of lagrange proof if you need to understand it but
We found Z(X) one calculation, we have the tree
We have the weights from our fast evaluation algorythm
all we need to do is combine efficinetly and we win

Z(X) = ZL(X)ZR(X)

for a singular index i

Z(X) / (x - xI) =  ZL/X-xI * ZR if Xi was in the left side
Likewise for the Right Side

ZR does not depend of i so we can Remove it from Summation

in the base case our ZR or ZL will literally simply
become the same as the denominator for that I value!

So we hence simply the entire equation down to just multiplying subtrees by each other
We can keep doing this logn times for each I term until the summation become a bunch of precalculated
treeRoots of ZLs and Zrs being multiplied * a weight that was calculated before
This algorythm is Genius. 
'''


def _fast_interpolate(weights, tree):
    if len(tree) == 1:
        return weights[0]
    
    W1 = weights[:len(tree.left)]
    W2 = weights[len(tree.left):]

    r0 = _fast_interpolate(W1, tree.left)
    r1 = _fast_interpolate(W2, tree.right)

    return r0 * tree.right.poly + r1 * tree.left.poly


print("Calculating Tree\n")

tree = compTree(X)

print("Tree Done. Calculating Weights\n")

weights = calcWeights(X, Y, tree)

print(" Weights calculated, fast interpolation time\n")

y = _fast_interpolate(weights, tree)

print("Done")

#Putting into image like original challenge suggested
open("output.bmp", "wb").write(bytearray(y.coefficients(sparse=False)[:-1]))
