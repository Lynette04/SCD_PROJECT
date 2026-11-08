from pytorch_training import count_params, count_trainable_params, evaluate, fit, load_model, plot_loss, save_model, summary, unfreeze_last_layers, set_random_seed
from step2_data_pipeline import get_datasets, get_class_weights

if __name__ == "__main__":
    initial_model = load_model("sickle_cell_model.pt") #load the saved model
    print(f"Loaded baseline model. Base model has {len(initial_model.base_model.layers)} layers.")
    summary(initial_model)


    #Training the model: fine-tuned version
    training_dataset, validation_dataset, test_dataset, class_names = get_datasets()
    class_weights, class_counts = get_class_weights() 
    
    #hyperparameters to test:
    layers_to_unfreeze = [8,14]
    learning_rates = [3e-6, 8e-6]
    
    best_recall = 0.0   
    best_val_loss = float("inf")
    best_config = None
    best_model = None
    best_history = None
    
    print("Let's start the fine tuning..\n")
    for num_layers in layers_to_unfreeze:
        for lr in learning_rates:
            print(f"\n--- TESTING: unfreeze_layers = {num_layers}, lr={lr} ---")
            #reset seed every run
            set_random_seed(42)
            #always reload fresh baseline weights from step3
            model = load_model("sickle_cell_model.pt")
            
            unfreeze_last_layers(model, num_layers)
    
            history = fit(
                model,
                training_dataset,
                validation_data = validation_dataset,
                epochs=20,
                class_weight = class_weights,
                #Here we use the Adam optimizer at a learning rate of 0.00001 so that small corrections are made each time the model is wrong.
                #It is smaller than the previous one of 0.001 because we are adjusting numbers in the base model which contain knowledge from 1.4million images.
                #So a smaller learning rate is needed to prevent spoiling the good knowledge of the base model.
                learning_rate = lr,
                weight_decay= 1e-4,
                patience = 6,
                verbose= True
            )
            
            #Best epoch's metrics
            best_epoch_idx = min(range(len(history)),key=lambda i: history[i]["val_loss"])
            best_epoch = history[best_epoch_idx]
            
            val_loss = best_epoch["val_loss"]
            val_recall= best_epoch["val_recall"]
            val_prec = best_epoch["val_precision"]
            print(f"Result -> Best Epoch({best_epoch_idx+1}): val_loss={val_loss:.4f}, val_recall={val_recall:.4f}, val_precision={val_prec:.4f}")
            
            #prioritizing recall(catching actual sickle cells) or lower loss
            is_better = False
            if val_recall > best_recall:
                is_better = True
            elif val_recall == best_recall and val_loss < best_val_loss:
                is_better= True
                
            if is_better:
                best_recall = val_recall
                best_val_loss = val_loss
                best_config = {"unfreeze_layers": num_layers, "lr": lr}
                best_model = model
                best_history = history
    print(f"\nWinner: {best_config} with Recall: {best_recall:.4f}")        
    plot_loss(best_history, "training_loss_finetune.png")

    #Using the test images to evaluate the fine-tuned model
    test_results = evaluate(best_model, test_dataset)
    for metric_name, value in test_results.items():
        print(f"{metric_name}:{value:.4f}")

    save_model(best_model, "Sickle_cell_fine_tuned-mode.pt")
    print("The model is saved in the desktop")
