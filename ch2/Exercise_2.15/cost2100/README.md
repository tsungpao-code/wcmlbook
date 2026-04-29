# Q7(a): COST2100 Channel Dataset Generation

This README explains how Q7(a) generates multiple channel datasets using the COST2100 channel model.  
The generated datasets are used later in Exercise 2.15 to evaluate the CSI reconstruction performance and generalization ability of CsiNet.

In Q7(a), the task is to generate **more than five different channel datasets**. Therefore, this implementation generates **six `.mat` files** by changing the user distribution.

The full COST2100 MATLAB source code is available at:
https://github.com/cost2100/cost2100

Only the modified Q7(a) generation script and generated datasets are included in this folder.
---

## What You Need to Do

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
```

The script should be placed in the COST2100 MATLAB folder:

```text
cost2100-master/cost2100-master/matlab/
```

Then run the following command in MATLAB:

```matlab
run q7a_generate_cost2100_datasets.m
```

After successful execution, the script creates:

```text
q7_generated_datasets/
```

This folder contains six generated `.mat` files.

---

### Part 2: Dataset Configuration

The original COST2100 demo usually generates one fixed channel scenario at a time. In this modified version, I added a dataset configuration list to automatically generate six datasets.

Each configuration defines:

- Dataset name
- COST2100 scenario
- Frequency range
- Base station position
- User distribution type
- User movement speed
- Generation radius

In this task, all datasets are based on the COST2100 `IndoorHall_5GHz` scenario, but different user distributions are used to create different channel statistics.

---

### Part 3: User Distribution Design

The main modification is the user-distribution generator. Instead of using only one user location setting, the script generates six different user distributions:

- **Uniform:** Users are uniformly distributed in the indoor region.
- **Center:** Users are concentrated near the base station.
- **Edge:** Users are located near the edge of the indoor region.
- **Hotspot:** Users are clustered around several hotspot areas.
- **Ring:** Users are distributed in a ring-shaped region.
- **Line:** Users are distributed along a line, similar to a corridor or walking route.

These different user distributions lead to different channel statistics, such as different path loss, delay structure, and spatial characteristics.

---

### Part 4: Generated Dataset Files

The script generates the following six datasets:

| Dataset File | Scenario | User Distribution | Description |
|---|---|---|---|
| `D1_indoor_uniform.mat` | `IndoorHall_5GHz` | Uniform | Baseline indoor dataset. Users are uniformly distributed. |
| `D2_indoor_center.mat` | `IndoorHall_5GHz` | Center | Users are closer to the base station, usually creating stronger channel conditions. |
| `D3_indoor_edge.mat` | `IndoorHall_5GHz` | Edge | Users are located near the boundary of the indoor area, creating a different location distribution. |
| `D4_indoor_hotspot.mat` | `IndoorHall_5GHz` | Hotspot | Users are grouped around several hotspot regions, representing non-uniform user distribution. |
| `D5_indoor_ring.mat` | `IndoorHall_5GHz` | Ring | Users are distributed around a ring-shaped area, used to test location-distribution shift. |
| `D6_indoor_line.mat` | `IndoorHall_5GHz` | Line | Users are distributed along a linear route, similar to a corridor or walking path. |

The reason six files are generated is that Q7(a) asks for “more than five different channel datasets.” Therefore, at least six datasets are required.

---

### Part 5: Saved Variables in Each `.mat` File

Each generated `.mat` file contains the following variables:

| Variable | Meaning |
|---|---|
| `H_complex` | Original complex CSI generated by the COST2100 channel model. |
| `H_norm` | Normalized complex CSI for more stable neural network training. |
| `H_real_imag` | CSI represented by two channels: real part and imaginary part. This format is suitable for CsiNet input. |
| `MSPos_all` | User positions used to generate the dataset. |
| `MSVelo_all` | User velocity vectors used in the COST2100 model. |
| `metadata` | Dataset information, including dataset name, scenario, user distribution, frequency range, BS position, and normalization factor. |

---

### Part 6: Main Difference from the Original COST2100 Demo

The original COST2100 `demo_model.m` mainly demonstrates how to generate one channel scenario. In this modified version, the script is extended for Q7(a) by adding:

- A dataset configuration list.
- A user-distribution generator.
- An automatic loop for generating six datasets.
- CSI normalization.
- Real/imaginary CSI formatting for neural network input.
- Metadata saving for each dataset.

Therefore, this script is not simply a copy of the original demo. It is modified to generate multiple datasets for testing CsiNet generalization.

---

### Part 7: Role in Exercise 2.15

The six datasets generated in Q7(a) are used for the next parts of Exercise 2.15:

| Part | Purpose |
|---|---|
| Q7(b) | Use a trained CsiNet model to reconstruct CSI on each dataset and calculate NMSE. |
| Q7(c) | Mix all six datasets to retrain CsiNet and compare the reconstruction performance with Q7(b). |

The purpose is to analyze whether CsiNet trained on one channel distribution can generalize to other unseen distributions, and whether mixed-dataset training can improve robustness in practical wireless systems.

---

## Summary

In Q7(a), I used the COST2100 channel model to generate six different channel datasets. All datasets are based on the `IndoorHall_5GHz` scenario, but each dataset uses a different user distribution. This design creates different channel statistics and allows later evaluation of CsiNet generalization.

The six generated datasets are:

```text
D1_indoor_uniform.mat
D2_indoor_center.mat
D3_indoor_edge.mat
D4_indoor_hotspot.mat
D5_indoor_ring.mat
D6_indoor_line.mat
```

These datasets are used as the basis for Q7(b) and Q7(c).
