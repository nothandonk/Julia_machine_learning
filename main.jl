using Pkg
Pkg.add("Flux")
Pkg.add("MLDataPattern")
Pkg.add("Images")
Pkg.add("FileIO")
using Flux
using Flux: onehotbatch, onecold, logitcrossentropy, @epochs
using Base.Iterators: repeated
using MLDataPattern: splitobs, stratifiedobs
using Images
using CSV
using DataFrames

# Load and preprocess the dataset
function load_images_labels(images_dir, labels_file)
    df = CSV.read(labels_file, DataFrame)
    images = []
    labels = []
    for row in eachrow(df)
        img_path = joinpath(images_dir, row[:filename])
        img = load(img_path)
        push!(images, float32(img))
        push!(labels, row[:label])
    end
    images, labels
end

# Map labels to integers for one-hot encoding
function map_labels_to_int(labels)
    unique_labels = unique(labels)
    label_map = Dict(label => i for (i, label) in enumerate(unique_labels))
    mapped_labels = [label_map[label] for label in labels]
    mapped_labels, label_map
end

images_dir = "C:/Users/Nothando/Documents/CSIR/Julia_practise/Julia_machine_learning/Img"
labels_file = "C:/Users/Nothando/Documents/CSIR/Julia_practise/Julia_machine_learning/english.csv"
images, labels = load_images_labels(images_dir, labels_file)
mapped_labels, label_map = map_labels_to_int(labels)
labels_onehot = onehotbatch(mapped_labels, 1:length(label_map))