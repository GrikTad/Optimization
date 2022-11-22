import numpy as np
    
def Animate2D(fun,x0_min,x0_max,x1_min,x1_max,x0_vals,x1_vals):
        from matplotlib import pyplot as plt
        from matplotlib import animation
        
        X,Y = np.meshgrid(np.linspace(x0_min,x0_max,100),np.linspace(x1_min,x1_max,100))
        zs = np.array(fun([np.ravel(X),np.ravel(Y)]))
        Z=zs.reshape(X.shape)
        fig,ax = plt.subplots(figsize=(7,7))
        ax.contour(X,Y,Z,100,cmap='jet')
        line, = ax.plot([],[],label='Path',lw=1.5) 
        point, = ax.plot([],[],'*',color='purple',markersize=4)
        value_display = ax.text(0.02,0.02,'',transform = ax.transAxes)
        
        def init_1():
            line.set_data([],[])
            point.set_data([],[]) 
            value_display.set_text('')
            
            return line,point,value_display
        def animate_1(i):
              line.set_data(x0_vals[:i],x1_vals[:i])
              point.set_data(x0_vals[i],x1_vals[i])
              
              return line, point, value_display
        
        ax.legend(loc=1)
        anim1 = animation.FuncAnimation(fig,animate_1,init_func=init_1,
                                        frames=len(x0_vals),interval=100,repeat_delay=60,blit=True)
        plt.show()
        