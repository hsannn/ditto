import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- Data Input ---
delta = [2.5, 3, 3.5, 4, 4.5, 5]

# # <MMW Bookreport>
# # p-value data
# p_values_llama3_1_8b = [6.83e-02, 1.99e-02, 4.36e-03, 1.83e-03, 7.97e-04, 1.83e-03]
# p_values_llama3_2_3b = [2.04e-06, 3.47e-10, 3.07e-12, 1.65e-17, 5.48e-17, 7.00e-20]

# # Perplexity data
# perplexity_llama3_1_8b = [3.271, 3.652, 3.937, 4.540, 4.182, 3.813]
# perplexity_llama3_2_3b = [3.786, 3.567, 3.226, 2.808, 2.441, 2.286]

# # <Dolly CW>
# # p-value data
# p_values_llama3_1_8b = [0.014, 4.36e-03, 3.09e-03, 4.32e-04, 4.83e-05, 3.75e-05]
# p_values_llama3_2_3b = [2.34e-03, 2.05e-05, 2.04e-06, 1.74e-08, 1.78e-07, 5.95e-10]

# # Perplexity data
# perplexity_llama3_1_8b = [3.864, 4.130, 4.312, 3.227, 3.281, 3.245]
# perplexity_llama3_2_3b = [5.051, 3.342, 4.330, 2.855, 2.951, 2.796]

# <SynthID Dolly CW>
# p-value data
p_values_llama3_1_8b = [2.33E-04, 6.74E-05, 1.41E-06, 6.84E-09, 7.10E-12, 4.84E-15]
p_values_llama3_2_3b = [1.76E-02, 4.96E-05, 8.50E-09, 1.75E-11, 8.12E-12, 6.50E-18]

# Perplexity data
perplexity_llama3_1_8b = [3.381, 3.249, 2.880, 2.817, 2.786, 2.386]
perplexity_llama3_2_3b = [3.799, 2.877, 2.675, 2.308, 2.876, 2.044]



fig, ax1 = plt.subplots(figsize=(12, 7))

ax1.set_xlabel('α', fontsize=35)
ax1.set_ylabel('p-value (log scale)', fontsize=35)
ax1.set_yscale('log')

line1 = ax1.plot(delta, p_values_llama3_1_8b, color='tab:blue', linestyle='-', marker='o', linewidth=6, markersize=20, label='p-value (llama3.1-8b)')[0]
line2 = ax1.plot(delta, p_values_llama3_2_3b, color='tab:pink', linestyle='-', marker='o', linewidth=6, markersize=20, label='p-value (llama3.2-3b)')[0]
ax1.tick_params(axis='y', labelsize=30)
ax1.tick_params(axis='x', labelsize=30)


ax2 = ax1.twinx()
ax2.set_ylabel('Perplexity', fontsize=35)

line3 = ax2.plot(delta, perplexity_llama3_1_8b, color='tab:blue', linestyle=':', marker='^', linewidth=6, markersize=20, label='Perplexity (llama3.1-8b)')[0]
line4 = ax2.plot(delta, perplexity_llama3_2_3b, color='tab:pink', linestyle=':', marker='^', linewidth=6, markersize=20, label='Perplexity (llama3.2-3b)')[0]

ax2.tick_params(axis='y', labelsize=30)




ax1.grid(True, which="both", linestyle='--', linewidth=0.5)

lines = [line1, line3, line2, line4]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=25)


plt.savefig('p-value_perplexity_vs_delta.png', dpi=300, bbox_inches='tight')
plt.close()