import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
import pandas as pd
import matplotlib
# Ensure Tkinter backend for Matplotlib before importing pyplot
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns

# Scientific computing imports
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, pairwise_distances

class GWOKMeansApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GWO-K-Means Clustering Optimizer")
        self.root.geometry("1200x800")
        
        # Data variables
        self.df = None
        self.X_preprocessed = None
        self.X_pca = None
        self.X_pca2 = None
        self.labels = None
        self.best_centroids = None
        self.results = {}
        
        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Data Upload and Configuration
        self.setup_data_tab()
        
        # Tab 2: Results and Visualization
        self.setup_results_tab()
        
        # Tab 3: Logs
        self.setup_logs_tab()
        
    def setup_data_tab(self):
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="Data & Configuration")
        
        # File upload section
        upload_frame = ttk.LabelFrame(self.data_frame, text="Data Upload", padding=10)
        upload_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(upload_frame, text="Upload Data File (Excel/CSV)", 
                  command=self.upload_file).pack(side=tk.LEFT, padx=5)
        
        self.file_label = ttk.Label(upload_frame, text="No file selected")
        self.file_label.pack(side=tk.LEFT, padx=10)
        
        # Data preview
        preview_frame = ttk.LabelFrame(self.data_frame, text="Data Preview", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Treeview for data preview
        self.tree = ttk.Treeview(preview_frame)
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        tree_scroll = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.tree.yview)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        
        # Configuration section
        config_frame = ttk.LabelFrame(self.data_frame, text="GWO-K-Means Configuration", padding=10)
        config_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Parameters
        params_frame = ttk.Frame(config_frame)
        params_frame.pack(fill=tk.X)
        
        # Left column
        left_col = ttk.Frame(params_frame)
        left_col.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Label(left_col, text="Number of Clusters (K):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.k_var = tk.IntVar(value=4)
        ttk.Spinbox(left_col, from_=2, to=20, textvariable=self.k_var, width=10).grid(row=0, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(left_col, text="Number of Wolves:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.wolves_var = tk.IntVar(value=25)
        ttk.Spinbox(left_col, from_=10, to=100, textvariable=self.wolves_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(left_col, text="Max Iterations:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.iterations_var = tk.IntVar(value=25)
        ttk.Spinbox(left_col, from_=10, to=200, textvariable=self.iterations_var, width=10).grid(row=2, column=1, sticky=tk.W, pady=2)
        
        # Right column
        right_col = ttk.Frame(params_frame)
        right_col.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        ttk.Label(right_col, text="Local Refine Steps:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.refine_var = tk.IntVar(value=1)
        ttk.Spinbox(right_col, from_=0, to=5, textvariable=self.refine_var, width=10).grid(row=0, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(right_col, text="Patience (Early Stop):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.patience_var = tk.IntVar(value=6)
        ttk.Spinbox(right_col, from_=3, to=20, textvariable=self.patience_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(right_col, text="Random Seed:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.seed_var = tk.IntVar(value=42)
        ttk.Spinbox(right_col, from_=1, to=1000, textvariable=self.seed_var, width=10).grid(row=2, column=1, sticky=tk.W, pady=2)
        
        # Run button
        run_frame = ttk.Frame(config_frame)
        run_frame.pack(fill=tk.X, pady=10)
        
        self.run_button = ttk.Button(run_frame, text="Run GWO-K-Means Optimization", 
                                   command=self.run_optimization)
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        self.progress = ttk.Progressbar(run_frame, mode='indeterminate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
    def setup_results_tab(self):
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Results & Visualization")
        
        # Results summary
        summary_frame = ttk.LabelFrame(self.results_frame, text="Optimization Results", padding=10)
        summary_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.results_text = tk.Text(summary_frame, height=8, width=80)
        self.results_text.pack(fill=tk.X)
        
        # Visualization area
        viz_frame = ttk.LabelFrame(self.results_frame, text="Visualizations", padding=10)
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Matplotlib canvas
        self.fig = Figure(figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Control buttons for different visualizations
        viz_controls = ttk.Frame(viz_frame)
        viz_controls.pack(fill=tk.X, pady=5)
        
        ttk.Button(viz_controls, text="Show PCA Scatter", 
                  command=self.show_pca_scatter).pack(side=tk.LEFT, padx=5)
        ttk.Button(viz_controls, text="Show Cluster Sizes", 
                  command=self.show_cluster_sizes).pack(side=tk.LEFT, padx=5)
        ttk.Button(viz_controls, text="Show Feature Heatmap", 
                  command=self.show_feature_heatmap).pack(side=tk.LEFT, padx=5)
        ttk.Button(viz_controls, text="Show Convergence", 
                  command=self.show_convergence).pack(side=tk.LEFT, padx=5)
        
    def setup_logs_tab(self):
        self.logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.logs_frame, text="Logs")
        
        # Log text area
        log_frame = ttk.LabelFrame(self.logs_frame, text="Optimization Logs", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=30, width=100)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Clear logs button
        ttk.Button(log_frame, text="Clear Logs", 
                  command=lambda: self.log_text.delete(1.0, tk.END)).pack(pady=5)
    
    def upload_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Load data
                if file_path.endswith(('.xlsx', '.xls')):
                    self.df = pd.read_excel(file_path)
                else:
                    self.df = pd.read_csv(file_path)
                
                self.file_label.config(text=f"Loaded: {os.path.basename(file_path)}")
                self.log(f"Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
                
                # Update preview
                self.update_data_preview()
                
                # Preprocess data
                self.preprocess_data()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
                self.log(f"Error loading file: {str(e)}")
    
    def update_data_preview(self):
        # Clear existing data
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        if self.df is not None:
            # Configure columns
            cols = list(self.df.columns)
            self.tree["columns"] = cols
            self.tree["show"] = "headings"
            
            # Configure column headings and widths
            for col in cols:
                self.tree.heading(col, text=col)
                self.tree.column(col, width=100)
            
            # Add data (first 100 rows)
            for idx, row in self.df.head(100).iterrows():
                values = [str(row[col]) for col in cols]
                self.tree.insert("", "end", values=values)
    
    def preprocess_data(self):
        if self.df is None:
            return
        
        try:
            self.log("Starting data preprocessing...")
            
            # Remove gender column if exists
            cols_drop = [c for c in ["gender"] if c in self.df.columns]
            if cols_drop:
                self.df = self.df.drop(columns=cols_drop)
                self.log(f"Dropped columns: {cols_drop}")
            
            # Handle binary columns (Ya/Tidak -> 1/0)
            binary_cols = []
            other_object_cols = []
            
            map_yes_no = {
                "ya": 1, "y": 1, "yes": 1, "true": 1, True: 1, 1: 1,
                "tidak": 0, "tdk": 0, "no": 0, "n": 0, "false": 0, False: 0, 0: 0
            }
            
            for col in self.df.columns:
                if self.df[col].dtype == object:
                    vals = self.df[col].dropna().astype(str).str.lower().str.strip().unique()
                    uniq = set(vals)
                    yes_set = {"ya", "y", "yes", "true"}
                    no_set = {"tidak", "tdk", "no", "n", "false"}
                    
                    if all((v in yes_set or v in no_set) for v in uniq) and len(uniq) <= 2:
                        binary_cols.append(col)
                    else:
                        other_object_cols.append(col)
            
            # Encode binary columns
            for col in binary_cols:
                self.df[col] = self.df[col].astype(str).str.lower().str.strip().map(map_yes_no)
            
            # Drop non-numeric, non-binary columns
            if other_object_cols:
                self.df = self.df.drop(columns=other_object_cols)
                self.log(f"Dropped non-numeric columns: {other_object_cols}")
            
            # Select numeric columns
            num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Imputation and scaling
            imputer = SimpleImputer(strategy="median")
            X_num = imputer.fit_transform(self.df[num_cols])
            
            scaler = MinMaxScaler()
            self.X_preprocessed = scaler.fit_transform(X_num)
            
            # PCA
            pca95 = PCA(n_components=0.95, svd_solver="full", random_state=42)
            self.X_pca = pca95.fit_transform(self.X_preprocessed)
            
            pca2 = PCA(n_components=2, random_state=42)
            self.X_pca2 = pca2.fit_transform(self.X_preprocessed)
            
            self.log(f"Preprocessing completed:")
            self.log(f"  - Features after preprocessing: {len(num_cols)}")
            self.log(f"  - PCA components (95% variance): {self.X_pca.shape[1]}")
            self.log(f"  - Data ready for clustering: {self.X_pca.shape}")
            
        except Exception as e:
            self.log(f"Error in preprocessing: {str(e)}")
            messagebox.showerror("Preprocessing Error", str(e))
    
    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
        self.root.update()
    
    def run_optimization(self):
        if self.X_pca is None:
            messagebox.showerror("Error", "Please upload and preprocess data first!")
            return
        
        # Disable run button and start progress
        self.run_button.config(state='disabled')
        self.progress.start()
        
        # Run optimization in separate thread
        threading.Thread(target=self._run_gwo_optimization, daemon=True).start()
    
    def _run_gwo_optimization(self):
        try:
            self.log("Starting GWO-K-Means optimization...")
            
            # Get parameters
            k = self.k_var.get()
            wolves = self.wolves_var.get()
            iters = self.iterations_var.get()
            local_refine = self.refine_var.get()
            patience = self.patience_var.get()
            seed = self.seed_var.get()
            
            self.log(f"Parameters: K={k}, Wolves={wolves}, Iterations={iters}")
            
            # Run GWO optimization
            best_centroids, history, iter_times, total_time, metrics_hist = self.gwo_only_with_history(
                self.X_pca, k, wolves, iters, seed, local_refine, patience
            )
            
            # Get best solution from history
            best_it, C_best, best_wcss = self.best_from_history(self.X_pca, history)
            self.labels = self.assign_labels(self.X_pca, C_best)
            self.best_centroids = C_best
            
            # Calculate final metrics
            sil = silhouette_score(self.X_pca, self.labels)
            ch = calinski_harabasz_score(self.X_pca, self.labels)
            db = davies_bouldin_score(self.X_pca, self.labels)
            
            # Store results
            self.results = {
                'best_iteration': best_it,
                'best_wcss': best_wcss,
                'silhouette': sil,
                'calinski_harabasz': ch,
                'davies_bouldin': db,
                'total_time': total_time,
                'history': history,
                'metrics_history': metrics_hist,
                'iter_times': iter_times
            }
            
            # Update results display
            self.root.after(0, self._update_results_display)
            
            self.log("GWO-K-Means optimization completed successfully!")
            
        except Exception as e:
            self.log(f"Error in optimization: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("Optimization Error", str(e)))
        
        finally:
            # Re-enable controls
            self.root.after(0, lambda: (
                self.run_button.config(state='normal'),
                self.progress.stop()
            ))
    
    def _update_results_display(self):
        results_summary = f"""
GWO-K-Means Optimization Results
================================

Best Iteration: {self.results['best_iteration']}
Best WCSS: {self.results['best_wcss']:.6f}

Clustering Metrics:
- Silhouette Score: {self.results['silhouette']:.6f}
- Calinski-Harabasz Score: {self.results['calinski_harabasz']:.6f}
- Davies-Bouldin Score: {self.results['davies_bouldin']:.6f}

Optimization Details:
- Total Runtime: {self.results['total_time']:.3f} seconds
- Number of Clusters: {self.k_var.get()}
- Number of Wolves: {self.wolves_var.get()}
- Max Iterations: {self.iterations_var.get()}

Data Information:
- Original Features: {self.X_preprocessed.shape[1]}
- PCA Components Used: {self.X_pca.shape[1]}
- Total Samples: {self.X_pca.shape[0]}
        """
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results_summary)
        
        # Switch to results tab
        self.notebook.select(1)
        
        # Show initial visualization
        self.show_pca_scatter()
    
    def show_pca_scatter(self):
        if self.labels is None:
            return
        
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        scatter = ax.scatter(self.X_pca2[:, 0], self.X_pca2[:, 1], c=self.labels, cmap='tab10')
        
        # Plot centroids (approximate in 2D space)
        centers2d = np.array([self.X_pca2[self.labels == c].mean(axis=0) for c in np.unique(self.labels)])
        ax.scatter(centers2d[:, 0], centers2d[:, 1], s=200, marker='X', c='red', edgecolor='black')
        
        for i, (cx, cy) in enumerate(centers2d):
            ax.text(cx, cy, f'C{i}', ha='center', va='center', fontweight='bold')
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('GWO-K-Means Clustering Results (PCA 2D Projection)')
        ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def show_cluster_sizes(self):
        if self.labels is None:
            return
        
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        unique, counts = np.unique(self.labels, return_counts=True)
        bars = ax.bar(unique, counts, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   str(count), ha='center', va='bottom')
        
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Cluster Size Distribution')
        ax.set_xticks(unique)
        ax.set_xticklabels([f'C{i}' for i in unique])
        ax.grid(axis='y', alpha=0.3)
        
        self.canvas.draw()
    
    def show_feature_heatmap(self):
        if self.labels is None or self.best_centroids is None:
            return
        
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Show first 6 PCA components
        n_components = min(6, self.best_centroids.shape[1])
        centroids_subset = self.best_centroids[:, :n_components]
        
        im = ax.imshow(centroids_subset, aspect='auto', cmap='viridis')
        self.fig.colorbar(im, ax=ax)
        
        ax.set_xticks(range(n_components))
        ax.set_xticklabels([f'PC{i+1}' for i in range(n_components)])
        ax.set_yticks(range(len(self.best_centroids)))
        ax.set_yticklabels([f'C{i}' for i in range(len(self.best_centroids))])
        ax.set_xlabel('Principal Components')
        ax.set_ylabel('Clusters')
        ax.set_title('Cluster Centroids Heatmap (PC1-PC6)')
        
        self.canvas.draw()
    
    def show_convergence(self):
        if 'history' not in self.results:
            return
        
        self.fig.clear()
        
        # Plot WCSS convergence
        ax1 = self.fig.add_subplot(211)
        iterations = [item[0] for item in self.results['history']]
        wcss_values = [self.wcss(self.X_pca, item[1]) for item in self.results['history']]
        
        ax1.plot(iterations, wcss_values, 'b-o', markersize=4)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('WCSS')
        ax1.set_title('GWO Convergence - WCSS')
        ax1.grid(True, alpha=0.3)
        
        # Plot runtime per iteration
        ax2 = self.fig.add_subplot(212)
        ax2.bar(range(1, len(self.results['iter_times'])+1), self.results['iter_times'], 
                color='orange', alpha=0.7)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Runtime per Iteration')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self.canvas.draw()
    
    # GWO Algorithm Implementation
    def assign_labels(self, X, C):
        return np.argmin(pairwise_distances(X, C, metric="euclidean"), axis=1)
    
    def wcss(self, X, C):
        lbl = self.assign_labels(X, C)
        d2 = pairwise_distances(X, C, metric="euclidean")**2
        return float(np.sum(d2[np.arange(X.shape[0]), lbl]))
    
    def repair_empty_clusters(self, X, C):
        k = C.shape[0]
        labels = self.assign_labels(X, C)
        for j in range(k):
            if np.sum(labels == j) == 0:
                D = pairwise_distances(X, C, metric="euclidean")
                idx = np.argmax(D.min(axis=1))
                C[j] = X[idx]
        return C
    
    def mini_lloyd_local(self, X, C0, steps=1):
        C = C0.copy()
        for _ in range(steps):
            C = self.repair_empty_clusters(X, C)
            lbl = self.assign_labels(X, C)
            for j in range(C.shape[0]):
                pts = X[lbl == j]
                if len(pts) > 0:
                    C[j] = pts.mean(axis=0)
        return self.wcss(X, C), C
    
    def gwo_only_with_history(self, X, k, wolves=24, iters=25, random_state=42, 
                             local_refine_steps=1, patience=6, reinit_rate=0.15):
        rng = np.random.RandomState(random_state)
        n, d = X.shape
        xmin, xmax = X.min(axis=0), X.max(axis=0)
        eps = 1e-12
        
        def rand_centroids():
            return xmin + rng.rand(k, d) * (xmax - xmin + eps)
        
        # Initialize population
        P = np.stack([rand_centroids() for _ in range(wolves)], axis=0)
        for i in range(wolves):
            _, P[i] = self.mini_lloyd_local(X, P[i], steps=1)
        
        fitness = np.array([self.wcss(X, P[i]) for i in range(wolves)])
        order = np.argsort(fitness)
        alpha = P[order[0]].copy()
        f_alpha = fitness[order[0]]
        beta = P[order[1]].copy() if wolves > 1 else alpha.copy()
        delta = P[order[2]].copy() if wolves > 2 else alpha.copy()
        
        history = []
        iter_times = []
        metrics_hist = []
        best_so_far = f_alpha
        no_improve = 0
        
        t0_global = time.perf_counter()
        
        for t in range(1, iters + 1):
            t_iter0 = time.perf_counter()
            a = 2 - 2 * (t - 1) / max(1, iters - 1)
            
            for i in range(wolves):
                r1, r2 = rng.rand(k, d), rng.rand(k, d)
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = np.abs(C1 * alpha - P[i])
                X1 = alpha - A1 * D_alpha
                
                r1, r2 = rng.rand(k, d), rng.rand(k, d)
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = np.abs(C2 * beta - P[i])
                X2 = beta - A2 * D_beta
                
                r1, r2 = rng.rand(k, d), rng.rand(k, d)
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = np.abs(C3 * delta - P[i])
                X3 = delta - A3 * D_delta
                
                P[i] = (X1 + X2 + X3) / 3.0
                P[i] = np.minimum(np.maximum(P[i], xmin), xmax)
                P[i] = self.repair_empty_clusters(X, P[i])
                
                if local_refine_steps > 0:
                    _, P[i] = self.mini_lloyd_local(X, P[i], steps=local_refine_steps)
            
            fitness = np.array([self.wcss(X, P[i]) for i in range(wolves)])
            order = np.argsort(fitness)
            alpha_new = P[order[0]].copy()
            f_alpha_new = fitness[order[0]]
            beta = P[order[1]].copy() if wolves > 1 else alpha_new.copy()
            delta = P[order[2]].copy() if wolves > 2 else alpha_new.copy()
            
            # Reinitialize worst wolves
            n_re = max(1, int(reinit_rate * wolves))
            for wi in order[-n_re:]:
                P[wi] = rand_centroids()
                _, P[wi] = self.mini_lloyd_local(X, P[wi], steps=1)
            
            alpha, f_alpha = alpha_new, f_alpha_new
            history.append((t, alpha.copy()))
            
            labels_alpha = self.assign_labels(X, alpha)
            try:
                sil = silhouette_score(X, labels_alpha)
                ch = calinski_harabasz_score(X, labels_alpha)
                db = davies_bouldin_score(X, labels_alpha)
            except:
                sil = ch = db = np.nan
            
            metrics_hist.append((sil, ch, db, f_alpha))
            
            t_iter = time.perf_counter() - t_iter0
            iter_times.append(t_iter)
            
            self.log(f"[GWO] iter={t:02d} | time={t_iter:.3f}s | best_WCSS={f_alpha:.6f}")
            self.log(f"      Silhouette: {sil:.6f} | CH: {ch:.6f} | DB: {db:.6f}")
            
            # Early stopping
            if f_alpha + 1e-9 < best_so_far:
                best_so_far = f_alpha
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    self.log(f"[GWO] Early stop (no improvement {patience} iters).")
                    break
        
        total_time = time.perf_counter() - t0_global
        return alpha, history, iter_times, total_time, metrics_hist
    
    def best_from_history(self, X, history):
        best = None
        for it, C in history:
            val = self.wcss(X, C)
            if best is None or val < best[2]:
                best = (it, C.copy(), val)
        return best


def main():
    root = tk.Tk()
    app = GWOKMeansApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()