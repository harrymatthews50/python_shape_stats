# python_shape_stats
A Python toolbox for statistical shape analysis of dense surface meshes. It allows:
  - Exploration of shape covariation within a sample using [principal components analysis](./demos/principal_components_analysis) and [determining the number of principal components](./demos/how_many_pcs.inpby)

<img src="./img/PC_1_2_3_4.gif" width="40%"> <img src="./img/PC_1_2_3_4.png" width="40%"> 

  - Assess covariation between structures with [two-block partial least-squares](./demos/2B_PLS.py)
  - Perform statistical hypothesis tests about the effect of variables on shape using a [partial least-squares regression model](./demos/pls_hypothesis_test.py)

## Explore covariation among structures with two-block partial least-squares
<img src="./docs/source/img/PLS_Dim1.gif" width="40%"> <img src="./docs/source/img/PLS_Dim1.png" width="40%"> 

[Read the cookbook](./docs/source/cookbooks/Two-Block_PLS/2B_PLS.ipynb)
