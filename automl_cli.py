"""
Dataset-Driven Automated Machine Learning CLI Tool

Main entry point for the AutoML CLI application
"""

import argparse
import sys
import os
from typing import Optional, List
from dataset_analyzer import DatasetAnalyzer
from model_selector import ModelSelector
from model_trainer import ModelTrainer
from preprocessor import DataPreprocessor
from hyperparameter_tuner import HyperparameterTuner
from visualizer import ModelVisualizer
from report_generator import ReportGenerator
from cli_utils import CLIFormatter, Colors, print_banner, ProgressTracker


def get_user_input(prompt: str, options: Optional[List[str]] = None, default: Optional[str] = None) -> str:
    """
    Get user input with validation
    
    Args:
        prompt: Question to ask user
        options: List of valid options (optional)
        default: Default value if user presses enter
        
    Returns:
        User's response
    """
    response = CLIFormatter.get_input(prompt, default)
    
    if not response and default:
        return default
    
    while True:
        if options:
            if response in options or response.lower() in [opt.lower() for opt in options]:
                return response
            CLIFormatter.print_warning(f"Invalid input. Choose from: {', '.join(options)}")
            response = CLIFormatter.get_input(prompt, default)
        else:
            if response:
                return response
            CLIFormatter.print_warning("Input cannot be empty")
            response = CLIFormatter.get_input(prompt, default)


def get_yes_no_input(prompt: str, default: str = 'n') -> bool:
    """
    Get yes/no input from user
    
    Args:
        prompt: Question to ask
        default: Default value ('y' or 'n')
        
    Returns:
        Boolean response
    """
    response = get_user_input(f"{prompt} (y/n)", options=['y', 'n', 'yes', 'no'], default=default)
    return response.lower() in ['y', 'yes']


def main():
    """Main application flow"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Automated Machine Learning CLI Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python automl_cli.py data/dataset.csv
  python automl_cli.py data/housing.csv --save-model
  python automl_cli.py data/large.csv --sample 50000 --quick
        """
    )
    parser.add_argument('dataset', type=str, help='Path to CSV dataset file')
    parser.add_argument('--save-model', action='store_true', 
                       help='Save the best model after training')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of dataset for testing (default: 0.2)')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel model training')
    parser.add_argument('--tune', action='store_true',
                       help='Enable hyperparameter tuning for best model')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations for best model')
    parser.add_argument('--report', action='store_true',
                       help='Generate comprehensive HTML report')
    parser.add_argument('--all', action='store_true',
                       help='Enable all features (tuning, visualizations, report)')
    parser.add_argument('--sample', type=int,
                       help='Use only N random rows (for large datasets)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: skip slow models (KNN, SVM)')
    parser.add_argument('--auto', action='store_true',
                       help='Fully automatic mode (no user prompts)')
    
    args = parser.parse_args()
    
    # Enable all features if --all flag is set
    if args.all:
        args.tune = True
        args.visualize = True
        args.report = True
        args.save_model = True
    
    # Print banner
    print_banner()
    
    # Calculate total steps
    TOTAL_STEPS = 7
    if args.tune:
        TOTAL_STEPS += 1
    if args.visualize:
        TOTAL_STEPS += 1
    if args.report:
        TOTAL_STEPS += 1
    if args.save_model and not args.report:  # Report already saves model
        TOTAL_STEPS += 1
    
    try:
        # Step 1: Load and analyze dataset
        CLIFormatter.print_step(1, TOTAL_STEPS, "Loading and Analyzing Dataset")
        analyzer = DatasetAnalyzer(args.dataset)
        analyzer.load_dataset()
        
        # Auto-sampling for large datasets
        if analyzer.df is not None:
            total_rows = len(analyzer.df)
            
            if args.sample and total_rows > args.sample:
                CLIFormatter.print_info(f"⚡ Sampling {args.sample:,} from {total_rows:,} rows...")
                analyzer.df = analyzer.df.sample(n=args.sample, random_state=42).reset_index(drop=True)
                print(f"{Colors.SUCCESS}✓ Using {len(analyzer.df):,} sampled rows{Colors.RESET}")
            elif not args.sample and total_rows > 100000:
                sample_size = 100000
                print(f"{Colors.WARNING}⚠ Large dataset: {total_rows:,} rows{Colors.RESET}")
                CLIFormatter.print_info(f"⚡ Auto-sampling to {sample_size:,} rows for speed")
                print(f"{Colors.DIM}   Use --sample N to customize{Colors.RESET}")
                analyzer.df = analyzer.df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        analyzer.analyze_dataset()
        analyzer.display_analysis()
        
        # Step 2: Get target column from user
        CLIFormatter.print_step(2, TOTAL_STEPS, "Target Column Selection")
        
        suggestions = analyzer.suggest_target_columns()
        print(f"\n{Colors.INFO}Suggested targets:{Colors.RESET} {Colors.HIGHLIGHT}{', '.join(suggestions)}{Colors.RESET}")
        
        if analyzer.df is not None:
            print(f"{Colors.DIM}All columns: {', '.join(analyzer.df.columns)}{Colors.RESET}")
        
        if args.auto and suggestions:
            target_column = suggestions[0]
            print(f"{Colors.SUCCESS}✓ Auto-selected target: {target_column}{Colors.RESET}")
        else:
            target_column = get_user_input("\nWhich column do you want to predict?")
        
        if analyzer.df is None or target_column not in analyzer.df.columns:
            CLIFormatter.print_error(f"Column '{target_column}' not found in dataset")
            return
        
        # Step 3: Identify problem type
        CLIFormatter.print_step(3, TOTAL_STEPS, "Problem Type Identification")
        
        problem_type, explanation = analyzer.identify_problem_type(target_column)
        print(f"\n{explanation}")
        
        if args.auto:
            interpretability = False
            print(f"\n{Colors.SUCCESS}✓ Auto-mode: Prioritizing accuracy{Colors.RESET}")
        else:
            interpretability = get_yes_no_input(
                "\nIs model interpretability a priority?",
                default='n'
            )
        
        # Determine dataset size
        n_rows = len(analyzer.df) if analyzer.df is not None else 0
        if n_rows < 1000:
            dataset_size = 'small'
        elif n_rows < 10000:
            dataset_size = 'medium'
        else:
            dataset_size = 'large'
        
        print(f"\n{Colors.INFO}Dataset size:{Colors.RESET} {Colors.BOLD}{dataset_size}{Colors.RESET} {Colors.DIM}({n_rows:,} rows){Colors.RESET}")
        
        # Step 5: Model selection
        CLIFormatter.print_step(5, TOTAL_STEPS, "Model Selection")
        
        selector = ModelSelector(problem_type)
        models = selector.select_models(
            interpretability_priority=interpretability,
            dataset_size=dataset_size
        )
        
        # Filter slow models in quick mode
        if args.quick and n_rows > 5000:
            slow_models = ['K-Nearest Neighbors', 'Support Vector']
            original_count = len(models)
            models = {k: v for k, v in models.items() 
                     if not any(slow in k for slow in slow_models)}
            skipped = original_count - len(models)
            if skipped > 0:
                print(f"{Colors.INFO}⚡ Quick mode: Skipped {skipped} slow model(s) (KNN/SVM){Colors.RESET}\n")
        
        selector.explain_model_selection()
        print(f"\n{Colors.SUCCESS}Selected {len(models)} models for training{Colors.RESET}")
        
        # Step 6: Data preprocessing
        CLIFormatter.print_step(6, TOTAL_STEPS, "Data Preprocessing")
        
        X, y = analyzer.get_feature_target_split(target_column)
        
        preprocessor = DataPreprocessor()
        X_processed, y_processed = preprocessor.fit_transform(X, y)
        
        preprocessor.get_preprocessing_summary()
        
        missing_info = analyzer.get_missing_value_info()
        if missing_info:
            print(f"\n{Colors.WARNING}Missing values handled:{Colors.RESET}")
            for col, info in list(missing_info.items())[:5]:  # Show first 5
                print(f"  {Colors.DIM}•{Colors.RESET} {col}: {info['count']} ({info['percentage']:.1f}%)")
        
        # Step 7: Model training and evaluation
        CLIFormatter.print_step(7, TOTAL_STEPS, "Model Training and Evaluation")
        
        trainer = ModelTrainer(problem_type)
        results = trainer.train_and_evaluate(
            models=models,
            X=X_processed,
            y=y_processed,
            test_size=args.test_size,
            parallel=not args.no_parallel
        )
        
        trainer.display_results()
        
        # Step 8: Hyperparameter Tuning (optional)
        current_step = 8
        tuned_params = {}
        if args.tune:
            CLIFormatter.print_step(current_step, TOTAL_STEPS, "Hyperparameter Tuning")
            current_step += 1
            
            tuner = HyperparameterTuner(problem_type, cv_folds=5)
            
            # Get original score
            best_model_name, best_model = trainer.get_best_model()
            if best_model_name:
                original_score = trainer.results[best_model_name].get(
                    'r2' if problem_type == 'regression' else 'f1', 0
                )
                
                # Tune the best model
                tuned_model, best_params, tuned_score = tuner.tune_model(
                    best_model_name, best_model, 
                    trainer.X_train, trainer.y_train,
                    use_randomized=True, n_iter=20
                )
                
                if best_params:
                    # Update best model with tuned version
                    trainer.best_model = tuned_model
                    trainer.trained_models[best_model_name] = tuned_model
                    tuned_params = best_params
                    
                    # Show tuning summary
                    print(tuner.get_tuning_summary(
                        best_model_name, original_score, 
                        tuned_score if tuned_score else original_score,
                        best_params
                    ))
        
        # Step 9: Generate Visualizations (optional)
        viz_paths = {}
        if args.visualize or args.report:
            CLIFormatter.print_step(current_step, TOTAL_STEPS, "Generating Visualizations")
            current_step += 1
            
            visualizer = ModelVisualizer(output_dir='visualizations')
            
            # Generate comparison plot for all models
            comp_path = visualizer.plot_model_comparison(trainer.results, problem_type)
            if comp_path:
                viz_paths['model_comparison'] = comp_path
            
            # Generate plots for best model
            if trainer.best_model_name:
                y_pred = trainer.predictions.get(trainer.best_model_name)
                y_proba = trainer.probabilities.get(trainer.best_model_name)
                
                feature_names = list(X_processed.columns) if hasattr(X_processed, 'columns') else None
                
                best_model_plots = visualizer.generate_all_plots(
                    trainer.best_model,
                    trainer.X_test,
                    trainer.y_test,
                    y_pred,
                    trainer.best_model_name,
                    problem_type,
                    feature_names=feature_names,
                    y_proba=y_proba
                )
                viz_paths.update(best_model_plots)
            
            if viz_paths:
                CLIFormatter.print_success(f"Saved {len(viz_paths)} visualizations to 'visualizations/' folder")
        
        # Step 10: Generate HTML Report (optional)
        if args.report:
            CLIFormatter.print_step(current_step, TOTAL_STEPS, "Generating Report")
            current_step += 1
            
            report_gen = ReportGenerator(output_dir='reports')
            
            # Prepare dataset info
            missing_info = analyzer.get_missing_value_info()
            total_missing = sum(info['count'] for info in missing_info.values()) if missing_info else 0
            
            dataset_info = {
                'path': args.dataset,
                'total_samples': len(analyzer.df) if analyzer.df is not None else 0,
                'total_features': len(X.columns) if hasattr(X, 'columns') else 0,
                'problem_type': problem_type.replace('_', ' ').title(),
                'target_column': target_column,
                'missing_values': total_missing
            }
            
            # Experiment configuration
            exp_config = {
                'test_size': args.test_size,
                'models_trained': len(trainer.results),
                'tuning_enabled': 'Yes' if args.tune else 'No',
                'interpretability': 'Yes' if interpretability else 'No'
            }
            
            # Generate report
            report_path = report_gen.generate_report(
                experiment_name=os.path.basename(args.dataset),
                dataset_info=dataset_info,
                results=trainer.results,
                best_model_name=trainer.best_model_name if trainer.best_model_name else 'None',
                best_model_metrics=trainer.results.get(trainer.best_model_name, {}) if trainer.best_model_name else {},
                problem_type=problem_type,
                config=exp_config,
                visualizations=viz_paths,
                tuned_params=tuned_params
            )
            
            CLIFormatter.print_success(f"Report saved: {report_path}")
            
            # Also save model if report is generated
            if trainer.best_model_name:
                model_name, model = trainer.get_best_model()
                filename = f"best_model_{problem_type}.joblib"
                trainer.save_model(filename)
        
        # Step 11: Save model if requested (and not already saved by report)
        elif args.save_model:
            CLIFormatter.print_step(current_step, TOTAL_STEPS, "Saving Model")
            
            model_name, model = trainer.get_best_model()
            filename = f"best_model_{problem_type}.joblib"
            trainer.save_model(filename)
        
        # Final summary
        print("\n")
        summary_items = {
            'Dataset': args.dataset,
            'Problem Type': problem_type.replace('_', ' ').title(),
            'Models Trained': len(trainer.results),
            'Best Model': trainer.best_model_name if trainer.best_model_name else 'None',
        }
        
        if args.tune and tuned_params:
            summary_items['Hyperparameter Tuning'] = 'Completed'
        
        if args.visualize or args.report:
            summary_items['Visualizations'] = f"{len(viz_paths)} plots generated"
        
        if args.report:
            summary_items['HTML Report'] = 'Generated'
        
        if args.save_model or args.report:
            summary_items['Model Saved'] = f"best_model_{problem_type}.joblib"
        
        CLIFormatter.print_summary_box(summary_items)
        
        if args.report:
            print(f"\n{Colors.HIGHLIGHT}📄 Open the HTML report in your browser to view detailed results!{Colors.RESET}")
        
        print(f"\n{Colors.SUCCESS}{Colors.BOLD}Thank you for using AutoML CLI! 🚀{Colors.RESET}\n")
        
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Process interrupted by user.{Colors.RESET}")
        sys.exit(0)
    except Exception as e:
        CLIFormatter.print_error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
