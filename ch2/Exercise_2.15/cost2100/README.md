##  Need to Do

| Step | Task | Details |
| :---: | :--- | :--- |
| 1 | **Download COST2100** | Download the COST2100 repository from GitHub and use the `matlab/` folder. The `cplusplus/` folder is not used in this task. |
| 2 | **Code Preparation** | Place `q7a_generate_cost2100_datasets.m` inside the COST2100 MATLAB directory. Make sure files such as `cost2100.m`, `create_IR_omni.m`, and `demo_model.m` are in the same MATLAB path. |
| 3 | **Run Dataset Generation** | Open MATLAB, change the current folder to the COST2100 `matlab/` directory, and run `q7a_generate_cost2100_datasets.m`. |
| 4 | **Check Generated Files** | After execution, a folder named `q7_generated_datasets/` will be created. It should contain six `.mat` channel datasets. |
| 5 | **Use for Q7(b)(c)** | Use the generated datasets to evaluate CsiNet reconstruction NMSE in Q7(b), and mix the datasets for CsiNet retraining in Q7(c). |

---

## File Structure

| File / Directory | Purpose |
|------|---------|
| `q7a_generate_cost2100_datasets.m` | Main MATLAB script for Q7(a). It automatically generates six COST2100 channel datasets by changing user distributions. |
| `cost2100.m` | Main COST2100 channel generation function. |
| `create_IR_omni.m` | Converts the COST2100 channel output into omnidirectional impulse response. |
| `demo_model.m` | Original COST2100 demo script. It is used only as a reference and is not directly copied. |
| `q7_generated_datasets/` | Output directory containing the six generated `.mat` channel datasets. |
| `D1_indoor_uniform.mat` | Dataset with uniformly distributed indoor users. |
| `D2_indoor_center.mat` | Dataset with users concentrated near the base station. |
| `D3_indoor_edge.mat` | Dataset with users located near the edge of the indoor area. |
| `D4_indoor_hotspot.mat` | Dataset with users clustered around hotspot regions. |
| `D5_indoor_ring.mat` | Dataset with users distributed in a ring-shaped region. |
| `D6_indoor_line.mat` | Dataset with users distributed along a line, similar to a corridor or walking route. |

---

## Detailed Task Breakdown

### Part 1: COST2100 Dataset Generation

This part uses the COST2100 channel model to generate more than five different channel datasets for Exercise 2.15 Q7(a). The purpose is to create multiple channel distributions so that the generalization ability of CsiNet can be evaluated in Q7(b) and Q7(c).

The MATLAB script used in this part is:

```text
q7a_generate_cost2100_datasets.m
