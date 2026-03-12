import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
import sympy as sp
import math
import warnings

warnings.filterwarnings("ignore")

x_sym = sp.Symbol('x')
max_terms = 50

x_vals = np.linspace(-10 * np.pi, 10 * np.pi, 1000)

functions_dict = {
    'sin(x) [a = π/2]':  {'expr': sp.sin(x_sym),    'a': np.pi / 2, 'ylim': (-3, 3)},
    'cos(x) [a = 0]':    {'expr': sp.cos(x_sym),    'a': 0,         'ylim': (-3, 3)},
    'exp(x) [a = 0]':    {'expr': sp.exp(x_sym),    'a': 0,         'ylim': (-5, 20)},
    'ln(x) [a = 1]':     {'expr': sp.log(x_sym),    'a': 1,         'ylim': (-5, 5)}
}

for name, data in functions_dict.items():
    expr = data['expr']
    a_point = data['a']

    f_lambdified = sp.lambdify(x_sym, expr, "numpy")
    data['y_true'] = f_lambdified(x_vals)

    coefficients = []
    deriv_expr = expr
    for n in range(max_terms):
        val_at_a = deriv_expr.subs(x_sym, a_point).evalf()
        c_n = float(val_at_a / sp.factorial(n))
        coefficients.append(c_n)
        deriv_expr = sp.diff(deriv_expr, x_sym)
        
    data['coefficients'] = coefficients

active_func = list(functions_dict.keys())[0]  
active_terms = 1

def calculate_taylor_y(func_name, num_terms):
    data = functions_dict[func_name]
    a_point = data['a']
    coefficients = data['coefficients']
    
    y_taylor = np.zeros_like(x_vals)
    for n in range(num_terms):
        if coefficients[n] != 0:
            y_taylor += coefficients[n] * (x_vals - a_point)**n
    return y_taylor


fig, ax = plt.subplots(figsize=(12, 7))
plt.subplots_adjust(bottom=0.25, left=0.25)

initial_data = functions_dict[active_func]
l1, = ax.plot(x_vals, initial_data['y_true'], label='True Function', color='blue', linewidth=2)
l2, = ax.plot(x_vals, calculate_taylor_y(active_func, active_terms), 
              label=f'Taylor Series ({active_terms} Terms)', color='red', linestyle='--', linewidth=2)

ax.set_ylim(initial_data['ylim'])
ax.set_xlim(x_vals[0], x_vals[-1])
ax.axhline(0, color='black', linewidth=0.8)
vline = ax.axvline(initial_data['a'], color='green', linestyle=':', label=f"Expansion Point: {initial_data['a']:.2f}")
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
title = ax.set_title(f"Taylor Expansion of {active_func}")

ax_radio = plt.axes([0.02, 0.05, 0.18, 0.15])
radio = RadioButtons(ax_radio, list(functions_dict.keys()))

ax_text = plt.axes([0.45, 0.08, 0.15, 0.05])
ax_text.axis('off')
text_terms = ax_text.text(0.5, 0.5, f"Terms: {active_terms}", fontsize=12, fontweight='bold', ha='center', va='center')

ax_down = plt.axes([0.40, 0.08, 0.05, 0.05])
btn_down = Button(ax_down, 'Down')

ax_up = plt.axes([0.60, 0.08, 0.05, 0.05])
btn_up = Button(ax_up, 'Up')

def update_plot():
    data = functions_dict[active_func]
    l1.set_ydata(data['y_true'])
    l2.set_ydata(calculate_taylor_y(active_func, active_terms))
    l2.set_label(f'Taylor Series ({active_terms} Terms)')

    ax.set_ylim(data['ylim'])
    vline.set_xdata([data['a'], data['a']])
    vline.set_label(f"Expansion Point: {data['a']:.2f}")
    
    ax.legend(loc='upper right')
    title.set_text(f"Taylor Expansion of {active_func}")
    text_terms.set_text(f"Terms: {active_terms}")
    
    fig.canvas.draw_idle()

def on_func_change(label):
    global active_func, active_terms
    active_func = label
    active_terms = 1 
    update_plot()

def on_up_click(event):
    global active_terms
    if active_terms < max_terms:
        active_terms += 1
        update_plot()

def on_down_click(event):
    global active_terms
    if active_terms > 1:
        active_terms -= 1
        update_plot()

radio.on_clicked(on_func_change)
btn_up.on_clicked(on_up_click)
btn_down.on_clicked(on_down_click)

plt.show()