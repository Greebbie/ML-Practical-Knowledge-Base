def get_content():
    return {
        "section": [
            {
                "title": "Optimization Algorithms: Overview",
                "description": """
                <p>Optimization algorithms are methods used to minimize the loss function of a neural network during training. They determine how parameters (weights and biases) are updated based on the gradient of the loss function.</p>
                <p>Key aspects of optimization algorithms:</p>
                <ul>
                    <li>Learning rate management</li>
                    <li>Convergence speed</li>
                    <li>Handling of local minima and saddle points</li>
                    <li>Robustness to noisy gradients</li>
                    <li>Memory requirements</li>
                </ul>
                """
            },
            {
                "title": "Gradient Descent",
                "description": """
                <p>Gradient Descent is the most basic optimization algorithm. It updates parameters in the opposite direction of the gradient of the loss function with respect to the parameters.</p>
                <p>Variants of gradient descent include:</p>
                <ul>
                    <li><strong>Batch Gradient Descent</strong>: Uses the entire dataset to compute gradients</li>
                    <li><strong>Stochastic Gradient Descent (SGD)</strong>: Uses a single sample at each iteration</li>
                    <li><strong>Mini-batch Gradient Descent</strong>: Uses a small batch of samples at each iteration</li>
                </ul>
                """,
                "formula": """
                $$\\theta = \\theta - \\eta \\nabla_\\theta J(\\theta)$$
                
                $$\\text{where } \\eta \\text{ is the learning rate and } \\nabla_\\theta J(\\theta) \\text{ is the gradient}$$
                """
            },
            {
                "title": "Momentum-Based Methods",
                "description": """
                <p>Momentum-based methods add a fraction of the previous update to the current update. This helps accelerate gradient descent by dampening oscillations and improving convergence.</p>
                <p>Key momentum-based methods:</p>
                <ul>
                    <li><strong>Classical Momentum</strong>: Adds a fraction of the previous update</li>
                    <li><strong>Nesterov Accelerated Gradient (NAG)</strong>: Looks ahead by computing the gradient at an estimated future position</li>
                </ul>
                """,
                "formula": """
                $$\\text{Classical Momentum:}$$
                $$v_t = \\gamma v_{t-1} + \\eta \\nabla_\\theta J(\\theta)$$
                $$\\theta = \\theta - v_t$$
                
                $$\\text{Nesterov Momentum:}$$
                $$v_t = \\gamma v_{t-1} + \\eta \\nabla_\\theta J(\\theta - \\gamma v_{t-1})$$
                $$\\theta = \\theta - v_t$$
                """
            },
            {
                "title": "Adaptive Learning Rate Methods",
                "description": """
                <p>Adaptive learning rate methods adjust the learning rate for each parameter based on historical gradient information. This allows for parameter-specific learning rates.</p>
                <p>Common adaptive methods include:</p>
                <ul>
                    <li><strong>AdaGrad</strong>: Adapts learning rates based on the historical squared gradients</li>
                    <li><strong>RMSProp</strong>: Extends AdaGrad with an exponentially decaying average</li>
                    <li><strong>Adam</strong>: Combines momentum and RMSProp ideas</li>
                    <li><strong>AdamW</strong>: Adam with decoupled weight decay</li>
                </ul>
                """,
                "formula": """
                $$\\text{AdaGrad:}$$
                $$G_t = G_{t-1} + (\\nabla_\\theta J(\\theta))^2$$
                $$\\theta = \\theta - \\frac{\\eta}{\\sqrt{G_t + \\epsilon}} \\nabla_\\theta J(\\theta)$$
                
                $$\\text{RMSProp:}$$
                $$E[g^2]_t = \\beta E[g^2]_{t-1} + (1-\\beta)(\\nabla_\\theta J(\\theta))^2$$
                $$\\theta = \\theta - \\frac{\\eta}{\\sqrt{E[g^2]_t + \\epsilon}} \\nabla_\\theta J(\\theta)$$
                
                $$\\text{Adam:}$$
                $$m_t = \\beta_1 m_{t-1} + (1-\\beta_1)\\nabla_\\theta J(\\theta)$$
                $$v_t = \\beta_2 v_{t-1} + (1-\\beta_2)(\\nabla_\\theta J(\\theta))^2$$
                $$\\hat{m}_t = \\frac{m_t}{1-\\beta_1^t}$$
                $$\\hat{v}_t = \\frac{v_t}{1-\\beta_2^t}$$
                $$\\theta = \\theta - \\frac{\\eta \\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon}$$
                """
            },
            {
                "title": "Second-Order Methods",
                "description": """
                <p>Second-order optimization methods use the Hessian matrix or approximations to it for faster convergence. They incorporate curvature information of the loss function.</p>
                <p>Examples include:</p>
                <ul>
                    <li><strong>Newton's Method</strong>: Uses the inverse Hessian matrix</li>
                    <li><strong>Quasi-Newton methods (BFGS, L-BFGS)</strong>: Approximate the Hessian matrix</li>
                    <li><strong>Conjugate Gradient</strong>: Builds conjugate directions iteratively</li>
                </ul>
                """,
                "formula": """
                $$\\text{Newton's Method:}$$
                $$\\theta = \\theta - [H_f(\\theta)]^{-1}\\nabla_\\theta J(\\theta)$$
                $$\\text{where } H_f(\\theta) \\text{ is the Hessian matrix of } J(\\theta)$$
                """
            },
            {
                "title": "Learning Rate Schedules",
                "description": """
                <p>Learning rate schedules adjust the learning rate during training to improve convergence.</p>
                <p>Common schedules include:</p>
                <ul>
                    <li><strong>Step Decay</strong>: Reduces the learning rate by a factor after a fixed number of epochs</li>
                    <li><strong>Exponential Decay</strong>: Exponentially decreases the learning rate</li>
                    <li><strong>Cosine Annealing</strong>: Uses a cosine function to gradually decrease the learning rate</li>
                    <li><strong>Cyclical Learning Rates</strong>: Cycles the learning rate between boundary values</li>
                    <li><strong>Warm Restarts</strong>: Periodically resets the learning rate to its initial value</li>
                </ul>
                """,
                "formula": """
                $$\\text{Step Decay: } \\eta_t = \\eta_0 \\times \\text{factor}^{\\lfloor t / \\text{drop-frequency} \\rfloor}$$
                
                $$\\text{Exponential Decay: } \\eta_t = \\eta_0 \\times e^{-kt}$$
                
                $$\\text{Cosine Annealing: } \\eta_t = \\eta_{\\text{min}} + \\frac{1}{2}(\\eta_{\\text{max}} - \\eta_{\\text{min}})(1 + \\cos(\\frac{t\\pi}{T}))$$
                """
            }
        ],
        "implementation": """
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

# Define a simple 2D function to optimize
def rosenbrock(x, y, a=1, b=100):
    # Rosenbrock function: f(x,y) = (a-x)^2 + b(y-x^2)^2
    return (a - x)**2 + b * (y - x**2)**2

# Gradient of the Rosenbrock function
def rosenbrock_grad(x, y, a=1, b=100):
    dx = -2 * (a - x) - 4 * b * x * (y - x**2)
    dy = 2 * b * (y - x**2)
    return np.array([dx, dy])

# Visualize the Rosenbrock function
def plot_rosenbrock():
    x = np.linspace(-2, 2, 200)
    y = np.linspace(-1, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)
    
    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis')
    plt.colorbar(label='f(x, y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Rosenbrock Function')
    
plot_rosenbrock()

# Implement optimization algorithms
def run_optimizers():
    # Starting point
    init_xy = np.array([-1.0, 1.0])
    
    # Parameters
    n_iters = 500
    learning_rate = 0.001
    
    # Store trajectories for different optimizers
    trajectories = {}
    
    # Gradient Descent
    xy = init_xy.copy()
    trajectory_gd = [xy.copy()]
    
    for i in range(n_iters):
        grad = rosenbrock_grad(xy[0], xy[1])
        xy = xy - learning_rate * grad
        trajectory_gd.append(xy.copy())
    
    trajectories['Gradient Descent'] = np.array(trajectory_gd)
    
    # Momentum
    xy = init_xy.copy()
    trajectory_momentum = [xy.copy()]
    velocity = np.zeros_like(xy)
    momentum = 0.9
    
    for i in range(n_iters):
        grad = rosenbrock_grad(xy[0], xy[1])
        velocity = momentum * velocity - learning_rate * grad
        xy = xy + velocity
        trajectory_momentum.append(xy.copy())
    
    trajectories['Momentum'] = np.array(trajectory_momentum)
    
    # RMSProp
    xy = init_xy.copy()
    trajectory_rmsprop = [xy.copy()]
    cache = np.zeros_like(xy)
    decay_rate = 0.99
    epsilon = 1e-8
    
    for i in range(n_iters):
        grad = rosenbrock_grad(xy[0], xy[1])
        cache = decay_rate * cache + (1 - decay_rate) * grad**2
        xy = xy - learning_rate * grad / (np.sqrt(cache) + epsilon)
        trajectory_rmsprop.append(xy.copy())
    
    trajectories['RMSProp'] = np.array(trajectory_rmsprop)
    
    # Adam
    xy = init_xy.copy()
    trajectory_adam = [xy.copy()]
    m = np.zeros_like(xy)
    v = np.zeros_like(xy)
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    
    for i in range(1, n_iters + 1):
        grad = rosenbrock_grad(xy[0], xy[1])
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**i)
        v_hat = v / (1 - beta2**i)
        xy = xy - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        trajectory_adam.append(xy.copy())
    
    trajectories['Adam'] = np.array(trajectory_adam)
    
    return trajectories

# Plot optimization trajectories
def plot_optimization_trajectories(trajectories):
    x = np.linspace(-2, 2, 200)
    y = np.linspace(-1, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)
    
    plt.figure(figsize=(12, 10))
    contour = plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis', alpha=0.5)
    
    colors = ['red', 'blue', 'green', 'purple']
    for i, (name, trajectory) in enumerate(trajectories.items()):
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'o-', color=colors[i], label=name, alpha=0.7, markersize=3)
    
    plt.plot(1, 1, 'ro', markersize=10, label='Global Minimum')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Optimization Trajectories on Rosenbrock Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
trajectories = run_optimizers()
plot_optimization_trajectories(trajectories)

# Implement learning rate schedules
def plot_learning_rate_schedules():
    n_epochs = 100
    init_lr = 0.1
    
    # Step decay
    step_size = 20
    gamma = 0.5
    step_decay = [init_lr * (gamma ** (epoch // step_size)) for epoch in range(n_epochs)]
    
    # Exponential decay
    exp_decay = [init_lr * np.exp(-0.01 * epoch) for epoch in range(n_epochs)]
    
    # Cosine annealing
    cosine = [0.001 + 0.5 * (init_lr - 0.001) * (1 + np.cos(np.pi * epoch / n_epochs)) for epoch in range(n_epochs)]
    
    # Cyclical (triangular)
    def triangular(epoch, base_lr=0.001, max_lr=0.1, step_size=20):
        cycle = np.floor(1 + epoch / (2 * step_size))
        x = np.abs(epoch / step_size - 2 * cycle + 1)
        return base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x))
    
    cyclical = [triangular(epoch) for epoch in range(n_epochs)]
    
    plt.figure(figsize=(12, 6))
    plt.plot(step_decay, label='Step Decay')
    plt.plot(exp_decay, label='Exponential Decay')
    plt.plot(cosine, label='Cosine Annealing')
    plt.plot(cyclical, label='Cyclical')
    
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedules')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
plot_learning_rate_schedules()

# PyTorch implementation examples
def pytorch_optimizer_examples():
    # Create a simple model and dummy data
    model = torch.nn.Linear(10, 1)
    data = torch.randn(100, 10)
    target = torch.randn(100, 1)
    
    # Define different optimizers
    sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    adam = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))
    rmsprop = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)
    adagrad = optim.Adagrad(model.parameters(), lr=0.01)
    
    # Example of using a learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(adam, step_size=10, gamma=0.5)
    
    # Training loop example (not executed, just for demonstration)
    for epoch in range(20):
        # Forward pass
        output = model(data)
        loss = torch.nn.functional.mse_loss(output, target)
        
        # Backward pass and optimize
        adam.zero_grad()
        loss.backward()
        adam.step()
        
        # Update learning rate
        scheduler.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item()}, LR: {scheduler.get_last_lr()[0]}")
""",
        "interview_examples": [
            {
                "title": "Comparing Optimization Algorithms",
                "description": "How would you compare different optimization algorithms and when should you use each?",
                "code": """
# Optimization algorithm comparison chart:

# 1. Vanilla SGD:
#    - Pros: Simple, robust, good regularization properties
#    - Cons: Slow convergence, sensitive to feature scaling, can get stuck
#    - Use when: Limited computation resources, simple problems, additional regularization needed

# 2. SGD with Momentum:
#    - Pros: Faster convergence, helps escape local minima, dampens oscillations
#    - Cons: Additional hyperparameter (momentum), requires more computation than vanilla SGD
#    - Use when: Training deep networks, dealing with noisy gradients

# 3. Nesterov Accelerated Gradient:
#    - Pros: Improved convergence over momentum, theoretically stronger guarantees
#    - Cons: More complex implementation, sensitive to learning rate
#    - Use when: Need more efficient acceleration than basic momentum

# 4. AdaGrad:
#    - Pros: Parameter-specific learning rates, good for sparse data
#    - Cons: Learning rate decreases too aggressively, can stop learning too early
#    - Use when: Dealing with sparse features or word embeddings

# 5. RMSProp:
#    - Pros: Solves AdaGrad's diminishing learning rate problem
#    - Cons: Not theoretically well understood
#    - Use when: Training recurrent neural networks, non-stationary objectives

# 6. Adam:
#    - Pros: Combines advantages of momentum and RMSProp, adaptive learning rates
#    - Cons: May converge to suboptimal solutions sometimes, can overfit
#    - Use when: Most deep learning applications, default choice for many practitioners

# 7. AdamW:
#    - Pros: Fixes weight decay regularization in Adam, often better generalization
#    - Cons: Additional hyperparameter tuning needed
#    - Use when: Training large networks where regularization is important

# 8. Second-order methods (L-BFGS):
#    - Pros: Fast convergence, fewer iterations needed
#    - Cons: Memory intensive, requires full batch computation
#    - Use when: Small to medium problems, when fast convergence is critical

def choose_optimizer(problem_characteristics):
    if problem_characteristics["non_stationary"]:
        if problem_characteristics["computational_resources"] == "high":
            return "Adam"
        else:
            return "RMSProp"
            
    elif problem_characteristics["sparse_data"]:
        return "AdaGrad"
        
    elif problem_characteristics["noisy_gradients"]:
        if problem_characteristics["need_regularization"]:
            return "AdamW"
        else:
            return "Adam"
            
    elif problem_characteristics["computation_per_iteration"] == "cheap":
        if problem_characteristics["memory_constraints"]:
            return "SGD with Momentum"
        else:
            return "Adam"
            
    elif problem_characteristics["batch_size"] == "full":
        return "L-BFGS"
        
    else:
        return "Adam"  # Safe default
"""
            },
            {
                "title": "Implementing Learning Rate Schedulers",
                "description": "How would you implement and use learning rate schedulers in a deep learning framework?",
                "code": """
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Custom learning rate scheduler implementation
class CustomCyclicalLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, base_lr, max_lr, step_size_up=2000, step_size_down=None, 
                 mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', last_epoch=-1):
        self.optimizer = optimizer
        self.base_lrs = [base_lr for _ in optimizer.param_groups]
        self.max_lrs = [max_lr for _ in optimizer.param_groups]
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down if step_size_down is not None else step_size_up
        self.mode = mode
        self.gamma = gamma
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode
        self.cycle_size = step_size_up + step_size_down
        self.step_counter = 0
        super(CustomCyclicalLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        cycle = np.floor(1 + self.step_counter / self.cycle_size)
        x = np.abs(self.step_counter / self.step_size_up - 2 * cycle + 1)
        
        if self.mode == 'triangular':
            scale_factor = 1.0
        elif self.mode == 'triangular2':
            scale_factor = 1.0 / (2.0 ** (cycle - 1))
        elif self.mode == 'exp_range':
            scale_factor = self.gamma ** self.step_counter
            
        self.step_counter += 1
        
        return [base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) * scale_factor
                for base_lr, max_lr in zip(self.base_lrs, self.max_lrs)]

# Different scheduler implementations using PyTorch
def visualize_schedulers():
    # Create a model
    model = torch.nn.Linear(10, 1)
    
    # Create optimizers for different schedulers
    opt1 = optim.SGD(model.parameters(), lr=0.1)
    opt2 = optim.SGD(model.parameters(), lr=0.1)
    opt3 = optim.SGD(model.parameters(), lr=0.1)
    opt4 = optim.SGD(model.parameters(), lr=0.1)
    opt5 = optim.SGD(model.parameters(), lr=0.1)
    
    # Create learning rate schedulers
    scheduler1 = optim.lr_scheduler.StepLR(opt1, step_size=20, gamma=0.5)
    scheduler2 = optim.lr_scheduler.ExponentialLR(opt2, gamma=0.97)
    scheduler3 = optim.lr_scheduler.CosineAnnealingLR(opt3, T_max=100, eta_min=0.001)
    scheduler4 = optim.lr_scheduler.ReduceLROnPlateau(opt4, mode='min', factor=0.5, patience=10)
    scheduler5 = optim.lr_scheduler.CyclicLR(opt5, base_lr=0.001, max_lr=0.1, step_size_up=10, mode='triangular')
    
    # Track learning rates for visualization
    lrs1, lrs2, lrs3, lrs4, lrs5 = [], [], [], [], []
    dummy_losses = [0.5 * (1 + np.cos(np.pi * i / 100)) for i in range(100)]
    
    # Simulate training loop
    for epoch in range(100):
        # Record learning rates
        lrs1.append(scheduler1.get_last_lr()[0])
        lrs2.append(scheduler2.get_last_lr()[0])
        lrs3.append(scheduler3.get_last_lr()[0])
        lrs5.append(scheduler5.get_last_lr()[0])
        
        # Special case for ReduceLROnPlateau which requires loss
        lrs4.append(opt4.param_groups[0]['lr'])
        
        # Step the schedulers
        scheduler1.step()
        scheduler2.step()
        scheduler3.step()
        scheduler5.step()
        
        # ReduceLROnPlateau needs loss value
        dummy_loss = dummy_losses[epoch]
        scheduler4.step(dummy_loss)
    
    # Plot learning rates
    plt.figure(figsize=(12, 6))
    plt.plot(lrs1, label='Step LR')
    plt.plot(lrs2, label='Exponential LR')
    plt.plot(lrs3, label='Cosine Annealing LR')
    plt.plot(lrs4, label='ReduceLROnPlateau')
    plt.plot(lrs5, label='Cyclic LR')
    
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Scheduler Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

# Usage in a training loop
def train_with_scheduler(model, train_loader, optimizer, scheduler, epochs=10):
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Print statistics
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Step the scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(running_loss)
        else:
            scheduler.step()
"""
            }
        ],
        "resources": [
            {"title": "An overview of gradient descent optimization algorithms", "url": "https://ruder.io/optimizing-gradient-descent/"},
            {"title": "Deep Learning Book - Optimization for Training Deep Models", "url": "https://www.deeplearningbook.org/contents/optimization.html"},
            {"title": "Optimization Methods for Large-Scale Machine Learning", "url": "https://arxiv.org/abs/1606.04838"}
        ],
        "related_topics": [
            "Gradient Descent", "Learning Rate Scheduling", "Neural Networks", "Loss Functions", "Backpropagation"
        ]
    } 