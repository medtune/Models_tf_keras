convolutions = [2,2,3,3,3]
nameMapping = {}
for i, j in range(1,6):
    newVariablescope = 'vgg_16/conv%d/%d'%(i,j)
    print(newVariablescope)
    oldVariablescope =  'vgg_16/conv%d/%d'%(i,j)
