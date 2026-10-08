from code.P1.micrograd.module import *
from code.P1.micrograd.trace_graph import *

x=[2.0, 3.0, -1.0]
n = MLP(3,[4,4,1])

xs = [[2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0]
      ]
ys = [1.0, -1.0, -1.0, 1.0]
for k in range(2000):
    ypred = [n(x) for x in xs]
    loss = sum([(yout - ygt) ** 2 for ygt, yout in zip(ys, ypred)])

    for p in n.parameters():
        p.grad = 0.0

    loss.backward()

    for p in n.parameters():
        p.data += -0.05 * p.grad

    print(k, loss.data)

dot = draw_dot(loss)
dot.render("loss_graph", view=True)