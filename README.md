# Reconhecimento de Imagens com AlexNet usando Keras

## Visão Geral

Este repositório contém o código e os recursos para desenvolver uma aplicação de reconhecimento de imagens utilizando o modelo AlexNet, implementado com o framework Keras. O modelo é treinado no conjunto de dados CIFAR-10, demonstrando a praticidade e eficiência do Keras na construção de modelos robustos e precisos de deep learning.

## Pré-requisitos

Certifique-se de ter os seguintes itens instalados em sua máquina:

    - Python 3.7 ou superior
    - pip (instalador de pacotes Python)
    - tensorflow 2.12

## Especificações do Servidor

O modelo foi executado em um servidor AWS g6.8xlarge com as seguintes especificações:

    vCPUs: 32
    Memória: 128 GB
    GPUs: 4 NVIDIA T4 Tensor Core GPUs
    Armazenamento: SSD baseado em NVMe
    Largura de banda de rede: Até 25 Gbps

Essas especificações proporcionaram o ambiente necessário para realizar o treinamento intensivo do modelo de deep learning com eficiência.

## Resultados

O modelo AlexNet treinado alcança uma acurácia de 66% no conjunto de teste do CIFAR-10 com um valor de perda de 1.2. Esses resultados indicam que, embora o modelo aprenda características significativas do conjunto de dados, ainda há potencial para melhorias.
