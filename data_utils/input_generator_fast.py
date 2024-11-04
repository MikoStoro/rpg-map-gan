import numpy as np




class Point:
    
    x = None
    y = None
    value = 0
    
    def __init__(self,value, x,y):
        self.x = x
        self.y=y
        self.value = value
        self.board = board
    
    def _append_point_if_legal(self, x, y):
        if x >=0 and x < x_dim and \
        y >= 0 and y < y_dim and \
        board[x][y] == 0:
            active_points.append( Point(self.value,x,y))
            
    def append_children(self):
        self._append_point_if_legal(self.x+1, self.y)
        self._append_point_if_legal(self.x-1, self.y)
        self._append_point_if_legal(self.x, self.y+1)
        self._append_point_if_legal(self.x, self.y-1)

num_of_points = 20
num_of_classes = 10
x_dim = 1024
y_dim = 1024
board = np.zeros((x_dim,y_dim))
active_points = []

def get_map(x = 128, y= 128, points = 5, classes = 5):
    global num_of_points
    global num_of_classes
    global x_dim
    global y_dim
    global board
    global active_points
    
    x_dim = x
    y_dim = y
    board = np.zeros((x_dim,y_dim))
    num_of_points = points
    num_of_classes = classes
    active_points = []
    
    for _ in range(num_of_points):
        x = np.random.randint(x_dim)
        y = np.random.randint(y_dim)
        if board[x][y] == 0:
            active_points.append(Point(np.random.randint(1,num_of_classes+1),x, y))

    while len(active_points) > 0:
        current_index = np.random.randint(len(active_points))
        current_point = active_points[current_index]
        x = current_point.x
        y = current_point.y
        
        if x >= 0 and x < x_dim and \
        y >= 0 and y < y_dim:
            board[x][y] = current_point.value
            current_point.append_children()
        active_points.pop(current_index)
        
    return board
 
 
def get_maps(n, x=128,y=128,points=5,classes=5):
    return [ get_map(x,y,points,classes) for _ in range(n) ]

def upsample_map(input_map, scale=2):
    return np.kron(input_map, np.ones((scale,scale)))

def upsample_maps(map_array, scale):
    return [  upsample_map(m, scale) for m in map_array  ]
    













