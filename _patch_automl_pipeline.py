"""
Patch script: replaces the Classical ML sub-tab in app.py with the
WatsonX AutoAI-style 7-step interactive pipeline.
"""
import re, os, sys

SRC = 'app.py'

OLD_MARKER_START = '    # --- SUB-TAB 1.1: CLASSICAL ML ---'
OLD_MARKER_END   = '    # --- SUB-TAB 1.2: COMPUTER VISION ---'

NEW_BLOCK = r'''    # --- SUB-TAB 1.1: CLASSICAL ML (WatsonX-Style Pipeline) ---
    with automl_tabs[0]:
        # ── Step definitions ──────────────────────────────────────────
        PIPELINE_STEPS = [
            ("📂", "Data"),
            ("🎯", "Task"),
            ("🤖", "Models"),
            ("⚡", "Optimize"),
            ("🛡️", "Validate"),
            ("🔧", "Advanced"),
            ("🚀", "Submit"),
        ]

        cur_step = st.session_state.get('automl_step', 0)
        render_pipeline_header(PIPELINE_STEPS, cur_step)

        # ── Shared state container ────────────────────────────────────
        cfg = st.session_state.get('automl_config', {})

        # ── Defaults that need to exist before any step ────────────────
        task                = cfg.get('task', 'classification')
        trainer_temp        = AutoMLTrainer(task_type=task)
        available_models    = trainer_temp.get_available_models()
        selected_models     = cfg.get('selected_models', None)
        training_preset     = cfg.get('training_preset', 'medium')
        training_strategy   = cfg.get('training_strategy', 'Automatic')
        manual_params       = cfg.get('manual_params', None)
        validation_strategy = cfg.get('validation_strategy', 'auto')
        validation_params   = cfg.get('validation_params', {})
        optimization_metric = cfg.get('optimization_metric', 'accuracy')
        selected_opt_mode   = cfg.get('optimization_mode', 'bayesian')
        n_trials            = cfg.get('n_trials', None)
        timeout_per_model   = cfg.get('timeout', None)
        total_time_budget   = cfg.get('time_budget', None)
        early_stopping      = cfg.get('early_stopping', 10)
        random_seed_config  = cfg.get('random_state', 42)
        ensemble_config     = cfg.get('ensemble_config', {})
        selected_nlp_cols   = cfg.get('selected_nlp_cols', [])
        nlp_config_automl   = cfg.get('nlp_config', {})
        enable_stability    = cfg.get('enable_stability', False)
        selected_stability_tests = cfg.get('stability_tests', [])
        target_pre          = cfg.get('target', None)
        date_col_pre        = cfg.get('date_col', None)
        selected_configs    = cfg.get('selected_configs', [])
        model_source        = cfg.get('model_source', 'Standard AutoML (Scikit-Learn/XGBoost/Transformers)')
        sample_df           = None
        embedding_model     = cfg.get('nlp_config', {}).get('embedding_model', 'all-MiniLM-L6-v2')

        # ════════════════════════════════════════════════════════════════
        # STEP 0 — Data Source
        # ════════════════════════════════════════════════════════════════
        if cur_step == 0:
            st.markdown("<h3 style='margin-bottom:4px;'>📂 Data Source</h3>", unsafe_allow_html=True)
            st.caption("Select one or more datasets from your Data Lake and configure the split strategy.")

            available_datasets = datalake.list_datasets()
            if not available_datasets:
                st.markdown("""
                <div class='ui-card' style='text-align:center;padding:32px;'>
                  <div style='font-size:2rem;'>🗄️</div>
                  <div style='font-weight:600;margin:8px 0;'>No datasets available</div>
                  <div style='color:#8b949e;font-size:0.85rem;'>Go to the <strong>Data</strong> tab to upload your first dataset.</div>
                </div>""", unsafe_allow_html=True)
            else:
                sel_ds_list = st.multiselect(
                    "Choose Datasets from Data Lake",
                    available_datasets,
                    default=cfg.get('ds_list', []),
                    key="wizard_ds_multi"
                )
                cfg['ds_list'] = sel_ds_list

                target_pre_w  = None
                date_col_pre_w= None

                if sel_ds_list:
                    # Sample preview from first dataset
                    try:
                        first_ds  = sel_ds_list[0]
                        first_ver = datalake.list_versions(first_ds)[0]
                        sample_df = datalake.load_version(first_ds, first_ver, nrows=5)

                        with st.expander("👁️ Data Preview", expanded=True):
                            st.dataframe(sample_df, use_container_width=True)

                        col_ta, col_dc = st.columns(2)
                        with col_ta:
                            # Task-aware target selector shown here for convenience
                            task_tmp = cfg.get('task', 'classification')
                            if task_tmp not in ["clustering", "anomaly_detection", "dimensionality_reduction"]:
                                default_target_idx = max(0, len(sample_df.columns) - 1)
                                target_pre_w = st.selectbox(
                                    "🎯 Target Column",
                                    sample_df.columns.tolist(),
                                    index=default_target_idx,
                                    key="wizard_target"
                                )
                                cfg['target'] = target_pre_w
                        with col_dc:
                            if task_tmp == 'time_series':
                                date_col_pre_w = st.selectbox("📅 Date Column", sample_df.columns, key="wizard_date")
                                cfg['date_col'] = date_col_pre_w
                    except Exception as e:
                        st.error(f"Error loading preview: {e}")

                    # Per-dataset version + split config
                    new_configs = []
                    
                    st.markdown("#### Configuração de Esquema (Schema)", unsafe_allow_html=True)
                    st.info("Ajuste os tipos de dados preenchidos automaticamente. Desmarque colunas que devem ser ignoradas no treinamento.")
                    
                    schema_df = pd.DataFrame({
                        "Incluir": [True] * len(sample_df.columns),
                        "Nome da coluna": sample_df.columns,
                        "Tipo": [str(t) for t in sample_df.dtypes],
                        "Valores de exemplo": [str(sample_df[c].iloc[0]) if len(sample_df) > 0 else "" for c in sample_df.columns]
                    })
                    
                    edited_schema = st.data_editor(
                        schema_df,
                        column_config={
                            "Incluir": st.column_config.CheckboxColumn("Incluir", help="Incluir no treinamento?", default=True),
                            "Tipo": st.column_config.SelectboxColumn("Tipo", help="Ocultar tipo do Pandas", options=["object", "int64", "float64", "bool", "datetime64[ns]"]),
                        },
                        disabled=["Nome da coluna", "Valores de exemplo"],
                        hide_index=True,
                        key="wizard_schema_editor"
                    )
                    cfg['schema_overrides'] = edited_schema.to_dict('records')
                    
                    st.markdown("#### Tipo de Divisão de Dados (Split)", unsafe_allow_html=True)
                    split_strategy = st.radio(
                        "Estratégia de Validação", 
                        ["Aleatório (Random)", "Cronológico (Chronological)", "Manual (Pre-defined split column)"],
                        horizontal=True,
                        key="wizard_split_strat"
                    )
                    cfg['split_strategy'] = split_strategy
                    
                    ds_cols = st.columns(min(len(sel_ds_list), 3))
                    for i, ds_name in enumerate(sel_ds_list):
                        with ds_cols[i % 3]:
                            versions = datalake.list_versions(ds_name)
                            ver = st.selectbox(f"📌 Version — {ds_name}", versions, key=f"wiz_ver_{ds_name}")
                            
                            # Render split progress bar logic based on images
                            split = st.slider(f"% Train — {ds_name}", 10, 100, 80, key=f"wiz_split_{ds_name}")
                            
                            st.markdown(f"**Split visual:** <span style='color:#2f80ed'>Training: {split}%</span> | <span style='color:#f59e0b'>Validation: {int((100-split)/2)}%</span> | <span style='color:#8b5cf6'>Testing: {100-split-int((100-split)/2)}%</span>", unsafe_allow_html=True)
                            
                            if split_strategy == "Cronológico (Chronological)":
                                time_col = st.selectbox(f"Coluna de Tempo p/ {ds_name}", sample_df.columns, key=f"wiz_time_{ds_name}")
                                new_configs.append({'name': ds_name, 'version': ver, 'split': split, 'time_column': time_col})
                            elif split_strategy == "Manual (Pre-defined split column)":
                                manual_col = st.selectbox(f"Coluna de Flag de Split p/ {ds_name}", sample_df.columns, key=f"wiz_manual_{ds_name}")
                                new_configs.append({'name': ds_name, 'version': ver, 'split': split, 'manual_split_column': manual_col})
                            else:
                                new_configs.append({'name': ds_name, 'version': ver, 'split': split})
                    cfg['selected_configs'] = new_configs

                    st.markdown('<br>', unsafe_allow_html=True)
                    data_ready = bool(sel_ds_list)
                    col_nav_fwd, _ = st.columns([1, 4])
                    with col_nav_fwd:
                        if st.button("Next: Define Task →", type="primary", key="step0_next", disabled=not data_ready):
                            st.session_state['automl_step'] = 1
                            st.session_state['automl_config'] = cfg
                            st.rerun()
                else:
                    st.info("ℹ️ Select at least one dataset above to continue.")

        # ════════════════════════════════════════════════════════════════
        # STEP 1 — Task Type & Learning Mode
        # ════════════════════════════════════════════════════════════════
        elif cur_step == 1:
            st.markdown("<h3 style='margin-bottom:4px;'>🎯 Task & Learning Type</h3>", unsafe_allow_html=True)
            st.caption("Select the ML task that best describes your problem.")

            SUPERVISED_TASKS = [
                ("classification", "🎯", "Classification", "Predict a category or label"),
                ("regression",     "📈", "Regression",     "Predict a continuous value"),
                ("time_series",    "⏳", "Time Series",    "Forecast future values from historical sequences"),
            ]
            UNSUP_TASKS = [
                ("clustering",               "🔵", "Clustering",               "Discover natural groups in data"),
                ("anomaly_detection",        "🚨", "Anomaly Detection",        "Identify unusual data points"),
                ("dimensionality_reduction", "🔻", "Dimensionality Reduction", "Compress features intelligently"),
            ]

            learn_type = st.radio(
                "Learning Paradigm",
                ["Supervised", "Unsupervised"],
                index=0 if cfg.get('task', 'classification') not in [t[0] for t in UNSUP_TASKS] else 1,
                horizontal=True,
                key="wiz_learn_type"
            )

            task_list = SUPERVISED_TASKS if learn_type == "Supervised" else UNSUP_TASKS
            current_task = cfg.get('task', task_list[0][0])
            if current_task not in [t[0] for t in task_list]:
                current_task = task_list[0][0]

            st.markdown('<br>', unsafe_allow_html=True)
            cols_t = st.columns(len(task_list))
            for i, (tid, ticon, tname, tdesc) in enumerate(task_list):
                with cols_t[i]:
                    is_sel = (tid == current_task)
                    border_style = "border: 2px solid #2f80ed; background: linear-gradient(135deg,rgba(47,128,237,0.1),rgba(139,92,246,0.1));" if is_sel else ""
                    st.markdown(f"""
                    <div class='task-card {"selected" if is_sel else ""}' style='{border_style}'>
                      <div class='task-icon'>{ticon}</div>
                      <div class='task-name'>{tname}</div>
                      <div class='task-desc'>{tdesc}</div>
                    </div>""", unsafe_allow_html=True)
                    if st.button(f"{'✓ Selected' if is_sel else 'Select'}", key=f"task_btn_{tid}"):
                        cfg['task'] = tid
                        st.session_state['automl_config'] = cfg
                        st.rerun()

            cfg['task'] = current_task
            task = current_task
            cfg['training_strategy'] = st.radio(
                "Hyperparameter Mode",
                ["Automatic", "Manual"],
                index=0 if cfg.get('training_strategy', 'Automatic') == 'Automatic' else 1,
                horizontal=True,
                help="Automatic: Optuna finds the best params. Manual: You define them.",
                key="wiz_hp_mode"
            )

            st.markdown('<br>', unsafe_allow_html=True)
            col_back, col_fwd, _ = st.columns([1, 1, 5])
            with col_back:
                if st.button("← Back", key="step1_back"):
                    st.session_state['automl_step'] = 0
                    st.session_state['automl_config'] = cfg
                    st.rerun()
            with col_fwd:
                if st.button("Next: Select Models →", type="primary", key="step1_next"):
                    st.session_state['automl_step'] = 2
                    st.session_state['automl_config'] = cfg
                    st.rerun()

        # ════════════════════════════════════════════════════════════════
        # STEP 2 — Model Selection
        # ════════════════════════════════════════════════════════════════
        elif cur_step == 2:
            task = cfg.get('task', 'classification')
            trainer_temp = AutoMLTrainer(task_type=task)
            available_models = trainer_temp.get_available_models()

            st.markdown("<h3 style='margin-bottom:4px;'>🤖 Model Selection</h3>", unsafe_allow_html=True)
            st.caption("Choose model source and which algorithms to include in the search.")

            model_source = st.radio(
                "Model Source",
                ["Standard AutoML (Scikit-Learn/XGBoost/Transformers)", "Model Registry (Registered)", "Local Upload (.pkl)"],
                index=["Standard AutoML (Scikit-Learn/XGBoost/Transformers)", "Model Registry (Registered)", "Local Upload (.pkl)"].index(cfg.get('model_source', 'Standard AutoML (Scikit-Learn/XGBoost/Transformers)')),
                horizontal=True,
                key="wiz_model_source"
            )
            cfg['model_source'] = model_source

            if model_source == "Standard AutoML (Scikit-Learn/XGBoost/Transformers)":
                mode_selection = st.radio(
                    "Selection Mode",
                    ["Automatic (Preset)", "Manual (Select)", "Custom Ensemble Builder"],
                    horizontal=True,
                    key="wiz_mode_sel"
                )
                cfg['mode_selection'] = mode_selection

                if mode_selection == "Manual (Select)":
                    sel_models = st.multiselect(
                        "Choose Models",
                        available_models,
                        default=cfg.get('selected_models', available_models[:2]) or available_models[:2],
                        key="wiz_sel_models"
                    )
                    cfg['selected_models'] = sel_models

                elif mode_selection == "Custom Ensemble Builder":
                    st.markdown("""
                    <div class='ui-card' style='padding:14px;margin-bottom:12px;'>
                      <div style='font-weight:600;margin-bottom:4px;'>🏗️ Ensemble Builder</div>
                      <div style='color:#8b949e;font-size:0.8rem;'>Combine multiple base models into a powerful ensemble.</div>
                    </div>""", unsafe_allow_html=True)
                    ensemble_type = st.selectbox("Ensemble Type", ["Voting", "Stacking"], key="wiz_ens_type")
                    base_candidates = [m for m in available_models if 'ensemble' not in m]
                    sel_base = st.multiselect("Base Estimators", base_candidates,
                        default=cfg.get('ensemble_config', {}).get('voting_estimators', base_candidates[:3]),
                        key="wiz_base_models")
                    if len(sel_base) < 2:
                        st.warning("⚠️ Select at least 2 base models.")

                    ens_cfg = {}
                    if ensemble_type == "Voting":
                        voting_type = st.selectbox("Voting Type", ["soft", "hard"] if task == "classification" else ["soft"], key="wiz_vote_type")
                        use_wts = st.checkbox("Weighted Voting", key="wiz_use_weights")
                        voting_weights = None
                        if use_wts:
                            wts_str = st.text_input("Weights (comma-separated)", value=",".join(["1.0"] * len(sel_base)), key="wiz_weights")
                            try:
                                voting_weights = [float(w.strip()) for w in wts_str.split(",")]
                            except:
                                st.error("Invalid weights format.")
                        ens_cfg = {'voting_estimators': sel_base, 'voting_type': voting_type, 'voting_weights': voting_weights}
                        cfg['selected_models'] = ['custom_voting']
                    else:  # Stacking
                        meta_candidates = ['logistic_regression', 'random_forest', 'xgboost', 'ridge', 'linear_regression']
                        if task == 'classification':
                            meta_candidates = [m for m in meta_candidates if m not in ['linear_regression', 'ridge']]
                        else:
                            meta_candidates = [m for m in meta_candidates if m not in ['logistic_regression']]
                        if not meta_candidates:
                            meta_candidates = ['random_forest']
                        final_est = st.selectbox("Meta-Model", meta_candidates, key="wiz_meta_model")
                        ens_cfg = {'stacking_estimators': sel_base, 'stacking_final_estimator': final_est}
                        cfg['selected_models'] = ['custom_stacking']

                    cfg['ensemble_config'] = ens_cfg
                else:
                    cfg['selected_models'] = None

            elif model_source == "Model Registry (Registered)":
                reg_models_list = get_registered_models()
                if reg_models_list:
                    base_name = st.selectbox("Registered Model", [m.name for m in reg_models_list], key="wiz_reg_model")
                    cfg['selected_models'] = [base_name]
                    st.info(f"Model **{base_name}** will be used as base for retraining.")
                else:
                    st.warning("No registered models found. Register a model in the Model Registry tab first.")
                    cfg['selected_models'] = None

            elif model_source == "Local Upload (.pkl)":
                uploaded_pkl = st.file_uploader("Upload .pkl file", type="pkl", key="wiz_pkl_upload")
                if uploaded_pkl:
                    cfg['selected_models'] = ["Uploaded_Model"]
                    st.success("Model loaded for retraining.")
                else:
                    cfg['selected_models'] = None

            st.markdown('<br>', unsafe_allow_html=True)
            col_back, col_fwd, _ = st.columns([1, 1, 5])
            with col_back:
                if st.button("← Back", key="step2_back"):
                    st.session_state['automl_step'] = 1
                    st.session_state['automl_config'] = cfg
                    st.rerun()
            with col_fwd:
                if st.button("Next: Optimization →", type="primary", key="step2_next"):
                    st.session_state['automl_step'] = 3
                    st.session_state['automl_config'] = cfg
                    st.rerun()

        # ════════════════════════════════════════════════════════════════
        # STEP 3 — Optimization Strategy
        # ════════════════════════════════════════════════════════════════
        elif cur_step == 3:
            task = cfg.get('task', 'classification')
            st.markdown("<h3 style='margin-bottom:4px;'>⚡ Optimization Strategy</h3>", unsafe_allow_html=True)
            st.caption("Define how AutoML searches for the best hyperparameters.")

            # Optimization mode as icon cards
            OPT_MODES = [
                ("bayesian",  "🧠", "Bayesian (Default)", "Efficient surrogate model-based search. Best for most use cases."),
                ("random",    "🎲", "Random Search",      "Randomly sample configurations. Fast and exploratory."),
                ("grid",      "📐", "Grid Search",        "Exhaustive search over all combinations. Slow but thorough."),
                ("hyperband", "⚡", "Hyperband",          "Early stopping of unpromising trials. Great for large datasets."),
            ]
            opt_mode_map = {"bayesian": "Bayesian Optimization (Default)", "random": "Random Search", "grid": "Grid Search", "hyperband": "Hyperband"}
            current_opt = cfg.get('optimization_mode', 'bayesian')

            cols_opt = st.columns(4)
            for i, (oid, oicon, oname, odesc) in enumerate(OPT_MODES):
                with cols_opt[i]:
                    is_sel = (oid == current_opt)
                    border_style = "border: 2px solid #2f80ed; background: linear-gradient(135deg,rgba(47,128,237,0.08),rgba(139,92,246,0.08));" if is_sel else ""
                    st.markdown(f"""
                    <div class='task-card' style='min-height:130px;{border_style}'>
                      <div class='task-icon'>{oicon}</div>
                      <div class='task-name'>{oname}</div>
                      <div class='task-desc'>{odesc}</div>
                    </div>""", unsafe_allow_html=True)
                    if st.button(f"{'✓' if is_sel else 'Select'}", key=f"opt_{oid}"):
                        cfg['optimization_mode'] = oid
                        st.session_state['automl_config'] = cfg
                        st.rerun()

            cfg['optimization_mode'] = current_opt if cfg.get('optimization_mode') == current_opt else cfg.get('optimization_mode', 'bayesian')

            st.markdown('<br>', unsafe_allow_html=True)

            # Training preset (slider)
            preset_labels = ["test", "fast", "medium", "high", "custom"]
            preset_descs  = {"test": "⚡ 1 trial for quick pipeline validation", "fast": "🚀 Fast search (~5 trials)",
                             "medium": "⚖️ Balanced speed/quality (default)", "high": "🎯 Exhaustive search",
                             "custom": "🔧 Manual trial/timeout configuration"}
            col_preset, col_metric = st.columns(2)
            with col_preset:
                training_preset = st.select_slider(
                    "Training Mode",
                    options=preset_labels,
                    value=cfg.get('training_preset', 'medium'),
                    key="wiz_preset"
                )
                cfg['training_preset'] = training_preset
                st.caption(preset_descs.get(training_preset, ""))

                if training_preset == "custom":
                    cfg['n_trials']       = st.number_input("Trials per model", 1, 1000, 20, key="wiz_trials")
                    cfg['timeout']        = st.number_input("Timeout per model (s)", 10, 7200, 600, key="wiz_timeout")
                    cfg['time_budget']    = st.number_input("Total time budget (s)", 60, 86400, 3600, key="wiz_total_time")
                    cfg['early_stopping'] = st.number_input("Early Stopping (rounds)", 0, 50, 7, key="wiz_es")
                    cfg['manual_params']  = {'max_iter': st.number_input("Max Iterations (max_iter)", 100, 100000, 1000, key="wiz_maxiter")}
                elif training_preset == "test":
                    st.warning("⚠️ TEST MODE: 1 trial, short timeout.")
                    cfg['n_trials'] = 1; cfg['timeout'] = 30; cfg['time_budget'] = 60; cfg['early_stopping'] = 1; cfg['manual_params'] = {}
                else:
                    cfg['n_trials'] = None; cfg['timeout'] = None; cfg['time_budget'] = None; cfg['early_stopping'] = 10; cfg['manual_params'] = {}

            with col_metric:
                metric_options = {
                    'classification': ['accuracy', 'f1', 'precision', 'recall', 'roc_auc'],
                    'regression': ['r2', 'rmse', 'mae'],
                    'clustering': ['silhouette'],
                    'time_series': ['rmse', 'mae', 'mape'],
                    'anomaly_detection': ['f1'],
                    'dimensionality_reduction': ['explained_variance']
                }
                metric_list = metric_options.get(task, ['accuracy'])
                cur_metric  = cfg.get('optimization_metric', metric_list[0])
                if cur_metric not in metric_list: cur_metric = metric_list[0]
                optimization_metric = st.selectbox(
                    "Optimization Metric",
                    metric_list,
                    index=metric_list.index(cur_metric),
                    key="wiz_metric"
                )
                cfg['optimization_metric'] = optimization_metric

                # Manual HP config if strategy == Manual
                if cfg.get('training_strategy') == 'Manual':
                    eff_models = cfg.get('selected_models') or available_models
                    trainer_tmp2 = AutoMLTrainer(task_type=task)
                    ref_model = st.selectbox("Configure Model", eff_models, key="wiz_manual_model")
                    schema = trainer_tmp2.get_model_params_schema(ref_model)
                    if schema:
                        st.markdown("**Manual Hyperparameters**")
                        mp = {}
                        cols_p = st.columns(3)
                        for pi, (p_name, p_cfg) in enumerate(schema.items()):
                            with cols_p[pi % 3]:
                                if p_cfg[0] == 'int':
                                    mp[p_name] = st.number_input(p_name, p_cfg[1], p_cfg[2], p_cfg[3], key=f"wiz_mp_{p_name}")
                                elif p_cfg[0] == 'float':
                                    mp[p_name] = st.number_input(p_name, p_cfg[1], p_cfg[2], p_cfg[3], format="%.4f", key=f"wiz_mp_{p_name}")
                                elif p_cfg[0] == 'list':
                                    options, p_def = p_cfg[1], p_cfg[2]
                                    mp[p_name] = st.selectbox(p_name, options, index=options.index(p_def) if p_def in options else 0, key=f"wiz_mp_{p_name}")
                        cfg['manual_params'] = mp

            st.markdown('<br>', unsafe_allow_html=True)
            col_back, col_fwd, _ = st.columns([1, 1, 5])
            with col_back:
                if st.button("← Back", key="step3_back"):
                    st.session_state['automl_step'] = 2
                    st.session_state['automl_config'] = cfg
                    st.rerun()
            with col_fwd:
                if st.button("Next: Validation →", type="primary", key="step3_next"):
                    st.session_state['automl_step'] = 4
                    st.session_state['automl_config'] = cfg
                    st.rerun()

        # ════════════════════════════════════════════════════════════════
        # STEP 4 — Validation Strategy
        # ════════════════════════════════════════════════════════════════
        elif cur_step == 4:
            task = cfg.get('task', 'classification')
            st.markdown("<h3 style='margin-bottom:4px;'>🛡️ Validation Strategy</h3>", unsafe_allow_html=True)
            st.caption("Choose how the model performance is evaluated during hyperparameter search.")

            validation_options = ["Automatic (Recommended)", "K-Fold Cross Validation", "Stratified K-Fold",
                                  "Holdout (Train/Test)", "Auto-Split (Optimized)", "Time Series Split"]

            if task == "time_series":
                val_strategy_ui = "Time Series Split"
                st.info("⏳ Time series must use temporal splitting.")
            elif task == "classification":
                val_strategy_ui = st.selectbox("Validation Method", validation_options,
                    index=max(0, validation_options.index(cfg.get('val_strategy_ui', 'Automatic (Recommended)'))
                              if cfg.get('val_strategy_ui') in validation_options else 0),
                    key="wiz_val_method")
            else:
                opts = [o for o in validation_options if o != "Stratified K-Fold"]
                v_default = cfg.get('val_strategy_ui', 'Automatic (Recommended)')
                if v_default not in opts: v_default = 'Automatic (Recommended)'
                val_strategy_ui = st.selectbox("Validation Method", opts,
                    index=opts.index(v_default), key="wiz_val_method_ns")

            cfg['val_strategy_ui'] = val_strategy_ui

            # Visual diagram
            if val_strategy_ui in ["K-Fold Cross Validation", "Stratified K-Fold"]:
                n_folds = st.slider("Number of Folds", 2, 20, cfg.get('validation_params', {}).get('folds', 5), key="wiz_folds")
                cfg['validation_params'] = {'folds': n_folds}
                cfg['validation_strategy'] = 'cv' if val_strategy_ui == "K-Fold Cross Validation" else 'stratified_cv'
                fold_pct = round(100 / n_folds)
                fold_bars = "".join([f"<div style='background:{'#2f80ed' if i==0 else '#1c2128'};border:1px solid #30363d;border-radius:4px;flex:1;height:20px;display:flex;align-items:center;justify-content:center;font-size:0.65rem;color:white;'>{('Val' if i==0 else 'Train')}</div>" for i in range(n_folds)])
                st.markdown(f"""
                <div style='background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px;margin:8px 0;'>
                  <div style='font-size:0.75rem;color:#8b949e;margin-bottom:6px;'>{n_folds}-Fold Cross Validation — each fold acts as validation once</div>
                  <div style='display:flex;gap:3px;'>{fold_bars}</div>
                </div>""", unsafe_allow_html=True)
            elif val_strategy_ui == "Holdout (Train/Test)":
                test_size_pct = st.slider("Test Split (%)", 10, 50, int(cfg.get('validation_params', {}).get('test_size', 0.2) * 100), key="wiz_holdout")
                cfg['validation_params'] = {'test_size': test_size_pct / 100.0}
                cfg['validation_strategy'] = 'holdout'
                train_w = 100 - test_size_pct
                st.markdown(f"""
                <div style='background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px;margin:8px 0;'>
                  <div style='font-size:0.75rem;color:#8b949e;margin-bottom:6px;'>Holdout Split: {train_w}% Train / {test_size_pct}% Test</div>
                  <div style='display:flex;border-radius:6px;overflow:hidden;height:24px;'>
                    <div style='flex:{train_w};background:#27ae60;display:flex;align-items:center;justify-content:center;font-size:0.7rem;color:white;font-weight:600;'>Train {train_w}%</div>
                    <div style='flex:{test_size_pct};background:#2f80ed;display:flex;align-items:center;justify-content:center;font-size:0.7rem;color:white;font-weight:600;'>Test {test_size_pct}%</div>
                  </div>
                </div>""", unsafe_allow_html=True)
            elif val_strategy_ui == "Time Series Split":
                n_splits = st.number_input("Temporal Splits", 2, 20, cfg.get('validation_params', {}).get('folds', 5), key="wiz_ts_splits")
                cfg['validation_params'] = {'folds': n_splits}
                cfg['validation_strategy'] = 'time_series_cv'
                ts_bars = "".join([f"<div style='background:{'#27ae60' if i<n_splits-1 else '#2f80ed'};border:1px solid #30363d;border-radius:3px;flex:1;height:20px;display:flex;align-items:center;justify-content:center;font-size:0.6rem;color:white;'>{'T' if i<n_splits-1 else 'V'}</div>" for i in range(n_splits)])
                st.markdown(f"""
                <div style='background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px;margin:8px 0;'>
                  <div style='font-size:0.75rem;color:#8b949e;margin-bottom:6px;'>Time Series Split — expanding window ({n_splits} splits)</div>
                  <div style='display:flex;gap:3px;'>{ts_bars}</div>
                </div>""", unsafe_allow_html=True)
            else:
                cfg['validation_strategy'] = 'auto' if val_strategy_ui == "Automatic (Recommended)" else 'auto_split'
                cfg['validation_params'] = {}
                st.info("✨ The system will automatically select the best validation strategy based on your dataset size and task type.")

            # NLP Configuration — render here if text columns exist
            st.markdown('<br>', unsafe_allow_html=True)
            ds_list_cur = cfg.get('ds_list', [])
            potential_nlp_cols = []
            if ds_list_cur:
                try:
                    fd = datalake.list_versions(ds_list_cur[0])[0]
                    sdf = datalake.load_version(ds_list_cur[0], fd, nrows=5)
                    potential_nlp_cols = sdf.select_dtypes(include=['object']).columns.tolist()
                except:
                    pass

            if potential_nlp_cols:
                with st.expander("🔤 NLP Configuration (Optional)"):
                    sel_nlp = st.multiselect("Text Columns", potential_nlp_cols,
                        default=cfg.get('selected_nlp_cols', []), key="wiz_nlp_cols")
                    cfg['selected_nlp_cols'] = sel_nlp
                    if sel_nlp:
                        col_n1, col_n2 = st.columns(2)
                        with col_n1:
                            vect = st.selectbox("Vectorizer", ["tfidf", "count", "embeddings"], key="wiz_vect")
                            if vect == "embeddings":
                                emb_model = st.selectbox("Embedding Model", ["all-MiniLM-L6-v2", "paraphrase-multilingual-MiniLM-L12-v2"], key="wiz_emb")
                            else:
                                ng_min, ng_max = st.slider("N-Gram Range", 1, 3, (1, 2), key="wiz_ngram")
                        with col_n2:
                            if vect != "embeddings":
                                rm_stop = st.checkbox("Remove Stopwords", value=True, key="wiz_stop")
                                lemma   = st.checkbox("Lemmatization", value=False, key="wiz_lemma")
                                max_feat= st.number_input("Max Features", 100, 50000, 5000, 500, key="wiz_maxfeat")
                            else:
                                st.info("Embeddings: dense fixed-length vectors.")
                        cfg['nlp_config'] = {
                            "vectorizer": vect,
                            "embedding_model": emb_model if vect == "embeddings" else None,
                            "ngram_range": (ng_min, ng_max) if vect != "embeddings" else (1, 1),
                            "stop_words": rm_stop if vect != "embeddings" else False,
                            "max_features": max_feat if vect != "embeddings" else 5000,
                            "lemmatization": lemma if vect != "embeddings" else False,
                        }

            st.markdown('<br>', unsafe_allow_html=True)
            col_back, col_fwd, _ = st.columns([1, 1, 5])
            with col_back:
                if st.button("← Back", key="step4_back"):
                    st.session_state['automl_step'] = 3
                    st.session_state['automl_config'] = cfg
                    st.rerun()
            with col_fwd:
                if st.button("Next: Advanced →", type="primary", key="step4_next"):
                    st.session_state['automl_step'] = 5
                    st.session_state['automl_config'] = cfg
                    st.rerun()

        # ════════════════════════════════════════════════════════════════
        # STEP 5 — Advanced Settings (Seed, Stability)
        # ════════════════════════════════════════════════════════════════
        elif cur_step == 5:
            task = cfg.get('task', 'classification')
            st.markdown("<h3 style='margin-bottom:4px;'>🔧 Advanced Settings</h3>", unsafe_allow_html=True)
            st.caption("Fine-tune reproducibility and post-training analysis options.")

            with st.expander("🌱 Reproducibility (Seed)", expanded=True):
                eff_models = cfg.get('selected_models') or (AutoMLTrainer(task_type=task).get_available_models())
                seed_mode = st.radio("Seed Mode",
                    ["Automatic (Different per model)", "Automatic (Same for all)", "Manual (Same for all)", "Manual (Different per model)"],
                    index=0, horizontal=True, key="wiz_seed_mode")
                rseed = 42
                if seed_mode == "Automatic (Different per model)":
                    rseed = {m: np.random.randint(0, 999999) for m in eff_models}
                    st.info(f"🎲 Random seeds generated per model.")
                elif seed_mode == "Automatic (Same for all)":
                    rseed = np.random.randint(0, 999999)
                    st.info(f"🎲 Same random seed for all: {rseed}")
                elif seed_mode == "Manual (Same for all)":
                    rseed = st.number_input("Global Seed", 0, 999999, cfg.get('random_state', 42) if isinstance(cfg.get('random_state'), int) else 42, key="wiz_seed_val")
                elif seed_mode == "Manual (Different per model)":
                    rseed = {}
                    sc = st.columns(min(len(eff_models), 3))
                    for si, sm in enumerate(eff_models):
                        with sc[si % 3]:
                            rseed[sm] = st.number_input(f"Seed: {sm}", 0, 999999, 42, key=f"wiz_seed_{sm}")
                cfg['random_state'] = rseed

            with st.expander("⚖️ Post-Training Stability Analysis (Optional)", expanded=False):
                enable_stab = st.checkbox("Run Stability Analysis after training",
                    value=cfg.get('enable_stability', False), key="wiz_enable_stab")
                cfg['enable_stability'] = enable_stab
                if enable_stab:
                    stab_opts = ["Data Variation Robustness", "Initialization Robustness", "Hyperparameter Sensitivity", "General Analysis"]
                    sel_stab = st.multiselect("Tests to Run", stab_opts,
                        default=cfg.get('stability_tests', ["General Analysis"]), key="wiz_stab_tests")
                    cfg['stability_tests'] = sel_stab
                    st.info("📊 Results will be saved to MLflow automatically.")

            st.markdown('<br>', unsafe_allow_html=True)
            col_back, col_fwd, _ = st.columns([1, 1, 5])
            with col_back:
                if st.button("← Back", key="step5_back"):
                    st.session_state['automl_step'] = 4
                    st.session_state['automl_config'] = cfg
                    st.rerun()
            with col_fwd:
                if st.button("Next: Review & Submit →", type="primary", key="step5_next"):
                    st.session_state['automl_step'] = 6
                    st.session_state['automl_config'] = cfg
                    st.rerun()

        # ════════════════════════════════════════════════════════════════
        # STEP 6 — Summary & Data Load & Submit
        # ════════════════════════════════════════════════════════════════
        elif cur_step == 6:
            task = cfg.get('task', 'classification')
            st.markdown("<h3 style='margin-bottom:4px;'>🚀 Review & Submit</h3>", unsafe_allow_html=True)
            st.caption("Review your configuration, load data, and launch the experiment.")

            # Config summary card
            opt_labels = {"bayesian": "Bayesian Optimization", "random": "Random Search", "grid": "Grid Search", "hyperband": "Hyperband"}
            sel_mods_disp = ", ".join(cfg.get('selected_models') or ["All (Automatic)"])
            st.markdown(f"""
            <div class='summary-card'>
              <div class='summary-row'><span class='summary-key'>📊 Dataset(s)</span><span class='summary-val'>{", ".join(cfg.get('ds_list', ["-"]))}</span></div>
              <div class='summary-row'><span class='summary-key'>🎯 Task</span><span class='summary-val'>{task.replace("_", " ").title()}</span></div>
              <div class='summary-row'><span class='summary-key'>🤖 Models</span><span class='summary-val'>{sel_mods_disp}</span></div>
              <div class='summary-row'><span class='summary-key'>⚡ Optimization</span><span class='summary-val'>{opt_labels.get(cfg.get('optimization_mode','bayesian'), 'Bayesian')}</span></div>
              <div class='summary-row'><span class='summary-key'>🏃 Preset</span><span class='summary-val'>{cfg.get('training_preset','medium').capitalize()}</span></div>
              <div class='summary-row'><span class='summary-key'>🛡️ Validation</span><span class='summary-val'>{cfg.get('val_strategy_ui', 'Automatic (Recommended)')}</span></div>
              <div class='summary-row'><span class='summary-key'>📈 Metric</span><span class='summary-val'>{cfg.get('optimization_metric','accuracy').upper()}</span></div>
              <div class='summary-row'><span class='summary-key'>⚖️ Stability</span><span class='summary-val'>{'✅ Enabled' if cfg.get('enable_stability') else '⬜ Disabled'}</span></div>
            </div>""", unsafe_allow_html=True)

            # --- Load Data ---
            st.markdown("### 📥 Load & Prepare Data")
            sel_cfgs = cfg.get('selected_configs', [])
            if not sel_cfgs:
                st.error("No datasets configured. Go back to Step 1.")
            else:
                if 'train_df' not in st.session_state or st.session_state.get('current_task') != task:
                    if st.button("📥 Load and Prepare Data", key="wiz_load_btn"):
                        with st.spinner("Loading & preparing data..."):
                            t_df, te_df = prepare_multi_dataset(
                                sel_cfgs, global_split=None,
                                task_type=task,
                                date_col=cfg.get('date_col'),
                                target_col=cfg.get('target')
                            )
                            st.session_state['train_df'] = t_df
                            st.session_state['test_df']  = te_df
                            st.session_state['current_task']        = task
                            st.session_state['date_col_active']     = cfg.get('date_col')
                            st.session_state['target_active']       = cfg.get('target')
                            st.session_state['n_trials_active']     = cfg.get('n_trials')
                            st.session_state['early_stopping_active'] = cfg.get('early_stopping', 10)
                        st.success(f"✅ Data loaded — {t_df.shape[0]:,} train rows × {t_df.shape[1]} cols")
                        st.rerun()
                else:
                    t_df = st.session_state['train_df']
                    st.success(f"✅ Data ready — {t_df.shape[0]:,} rows × {t_df.shape[1]} cols")

            # Final target / time-series config (only if data is loaded)
            target = None
            forecast_horizon, freq = 1, "D"

            if 'train_df' in st.session_state and st.session_state.get('current_task') == task:
                train_df = st.session_state['train_df']
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    if task not in ["clustering", "anomaly_detection", "dimensionality_reduction"]:
                        act_target = st.session_state.get('target_active')
                        if act_target and act_target in train_df.columns:
                            target = act_target
                            st.info(f"🎯 Target: **{target}**")
                        else:
                            target = st.selectbox("🎯 Select Target", train_df.columns, key="wiz_final_target")
                with col_f2:
                    if task == "time_series":
                        freq = st.selectbox("⏱️ Frequency", ["Minutes","Hours","Days","Weeks","Months","Years"], key="wiz_freq")
                        forecast_horizon = st.number_input("🔮 Horizon", 1, 100, 7, key="wiz_horizon")

                # --- Submit ---
                st.divider()
                exp_name_default = f"{sel_cfgs[0]['name'] if sel_cfgs else 'AutoML'}_{task}_{time.strftime('%Y%m%d_%H%M%S')}"
                exp_name = st.text_input("Experiment Name", value=exp_name_default, key="wiz_exp_name")
                clean_exp_name = "".join(c for c in exp_name if ord(c) < 128) or "AutoML_Experiment"

                col_back2, col_sub, _ = st.columns([1, 2, 4])
                with col_back2:
                    if st.button("← Back", key="step6_back"):
                        st.session_state['automl_step'] = 5
                        st.session_state['automl_config'] = cfg
                        st.rerun()
                with col_sub:
                    if st.button("🚀 Submit Experiment", key="wiz_submit_btn", type="primary"):
                        real_sel_models = cfg.get('selected_models')
                        real_ens_cfg    = cfg.get('ensemble_config', {})
                        real_opt_mode   = cfg.get('optimization_mode', 'bayesian')
                        real_preset     = cfg.get('training_preset', 'medium')
                        real_n_trials   = cfg.get('n_trials')
                        real_timeout    = cfg.get('timeout')
                        real_time_bud   = cfg.get('time_budget')
                        real_es         = cfg.get('early_stopping', 10)
                        real_mp         = cfg.get('manual_params')
                        real_val_strat  = cfg.get('validation_strategy', 'auto')
                        real_val_params = cfg.get('validation_params', {})
                        real_seed       = cfg.get('random_state', 42)
                        real_nlp_cols   = cfg.get('selected_nlp_cols', [])
                        real_nlp_cfg    = cfg.get('nlp_config', {})
                        real_stab       = cfg.get('enable_stability', False)
                        real_stab_tests = cfg.get('stability_tests', [])

                        job_config = {
                            'train_df': st.session_state.get('train_df'),
                            'test_df':  st.session_state.get('test_df'),
                            'target': target,
                            'task': task,
                            'date_col': cfg.get('date_col'),
                            'forecast_horizon': forecast_horizon,
                            'selected_nlp_cols': real_nlp_cols,
                            'nlp_config': real_nlp_cfg,
                            'preset': real_preset,
                            'n_trials': real_n_trials,
                            'timeout': real_timeout,
                            'time_budget': real_time_bud,
                            'selected_models': real_sel_models,
                            'manual_params': real_mp,
                            'experiment_name': clean_exp_name,
                            'random_state': real_seed,
                            'validation_strategy': real_val_strat,
                            'validation_params': real_val_params,
                            'ensemble_config': real_ens_cfg,
                            'optimization_mode': real_opt_mode,
                            'optimization_metric': cfg.get('optimization_metric', 'accuracy'),
                            'target_metric_name': cfg.get('optimization_metric', 'accuracy').upper(),
                            'early_stopping': real_es,
                            'stability_config': {'tests': real_stab_tests, 'n_iterations': 3} if real_stab else None,
                            'mlflow_tracking_uri': mlflow.get_tracking_uri(),
                            'dagshub_user': os.environ.get('MLFLOW_TRACKING_USERNAME'),
                            'dagshub_token': os.environ.get('MLFLOW_TRACKING_PASSWORD'),
                        }
                        jm = st.session_state['job_manager']
                        job_id = jm.submit_job(job_config, name=clean_exp_name)
                        st.success(f"✅ Experiment **{clean_exp_name}** submitted! (ID: `{job_id}`)")
                        st.info("📊 View live progress in the **Experiments** tab.")
                        st.balloons()
                        # Reset wizard for next experiment
                        st.session_state['automl_step'] = 0
                        st.session_state['automl_config'] = {}
            else:
                col_back3, _ = st.columns([1, 6])
                with col_back3:
                    if st.button("← Back", key="step6_back_nodata"):
                        st.session_state['automl_step'] = 5
                        st.session_state['automl_config'] = cfg
                        st.rerun()

    # --- SUB-TAB 1.2: COMPUTER VISION ---
'''

with open(SRC, encoding='utf-8') as f:
    content = f.read()

start_marker = '    # --- SUB-TAB 1.1: CLASSICAL ML ---'
end_marker   = '    # --- SUB-TAB 1.2: COMPUTER VISION ---'

i_start = content.find(start_marker)
i_end   = content.find(end_marker)

if i_start == -1 or i_end == -1:
    print(f"ERROR: markers not found. start={i_start}, end={i_end}")
    sys.exit(1)

new_content = content[:i_start] + NEW_BLOCK + content[i_end:]

with open(SRC, 'w', encoding='utf-8') as f:
    f.write(new_content)

print(f"OK — replaced {i_end - i_start} chars ({content[i_start:i_end].count(chr(10))} lines) with new pipeline wizard.")
print(f"New file size: {len(new_content)} chars, {new_content.count(chr(10))} lines")
