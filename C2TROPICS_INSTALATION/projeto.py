#!/usr/bin/env python
# coding: utf-8
# ######################
# Import libraries
######################
# 
import streamlit as st
import streamlit.components.v1 as stc
import numpy as np
# File Processing Pkgs
import pandas as pd
import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image

import os
import joblib

import numpy as np
import pandas as pd
from itertools import combinations
import mols2grid
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components



def main():
    # st.title("Molecular Desreplication - APP")
    # menu = ["Home", "Deep Learning", "Classificação", "Molecular Docking", "About"]
    # menu = ["Home", "Deep Learning",  "Docking", "Data Molecular Dynamics", "About"]
    menu = ["Home", 'Aprendizado de Máquina', "Docking",  "About"]
    import streamlit as st
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        from PIL import Image
        image = Image.open('capa_app.png')
        image = image.resize((800, 600))
        st.image(image, use_container_width=True)
        image = Image.open('cheic-aba.png')
        st.sidebar.image(image, use_container_width=True)

   

    elif choice == ('Aprendizado de Máquina'):
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        import pandas as pd
        import streamlit as st
        from rdkit.Chem import Draw
        import io

        def process_smiles(smiles_list, model, feature_list):

            molecules = []
            descriptors_list = []

            # Processar todos os SMILES para calcular os descritores
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    molecules.append(mol)
                    descriptors = [
                        getattr(Descriptors, desc)(mol) if getattr(Descriptors, desc, None) else None
                        for desc in feature_list
                    ]
                    descriptors_list.append(descriptors)
                else:
                    molecules.append(None)
                    descriptors_list.append([None] * len(feature_list))  # Molécula inválida

            # Criar um DataFrame com os descritores calculados
            features_df = pd.DataFrame(descriptors_list, columns=feature_list)
            
            predictions = model.predict(features_df)

            # Criar um DataFrame com os SMILES, descritores e predições
            results = pd.DataFrame({
                'SMILES': smiles_list,
                **{desc: features_df[desc] for desc in feature_list},
                'Prediction': predictions
            })

            st.write("resultados:")
            st.write(results)
            
            return results, molecules, predictions


        # Carregar o modelo treinado
        def load_model():
            return joblib.load("model_with_features.pkl")  # Substitua pelo caminho correto do modelo

        # Streamlit UI
        st.title("QUÍMICA COMPUTACIONAL NO COMBATE DA DENGUE")
        st.markdown("""
            Insira um ou mais SMILES (separados por vírgula) para gerar os descritores e fazer a previsão.
        """)

        # Entrada do usuário para os SMILES
        smiles_input = st.text_area("Digite SMILES")
        # Carregar modelo e features
        modelo_salvo = load_model()
        model = modelo_salvo["model"]
        feature_list = modelo_salvo["features"]

        # Lista de SMILES do usuário
        smiles_list = [smiles.strip() for smiles in smiles_input.split(",") if smiles.strip()]

        # Botão para carregar um arquivo Excel com SMILES
        uploaded_file = st.file_uploader("Carregar arquivo Excel com SMILES", type="xlsx")

        # Processar o arquivo carregado
        if st.button("Processar"):
            if uploaded_file:
                # Carregar e processar o arquivo Excel
                df = pd.read_excel(uploaded_file)
                smiles_list.extend(df["smiles"].dropna().tolist())

            if smiles_list:
                results, molecules, predictions = process_smiles(smiles_list, model, feature_list)

                # Exibir resultados
                for mol, pred in zip(molecules, predictions):
                    if mol:
                        img = Draw.MolToImage(mol, size=(300, 300))  # Diminuir o tamanho da imagem (300x300)
                        st.image(img, caption=f"Predição: {pred}", width=300)  # Exibe a imagem com o novo tamanho
                        
                        # Aumentar o tamanho da fonte da predição
                        if pred == "ativa":
                            st.success(f"<h2 style='font-size:24px; color: green;'>Predição: {pred}</h2>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<h3 style='font-size:24px;'>Predição: {pred}</h3>", unsafe_allow_html=True)
                    else:
                        st.warning("Molécula inválida.")

                # Gerar CSV para download
                csv_buffer = io.StringIO()
                results.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()

                # Botão de download
                st.download_button(
                    label="Baixar resultados como CSV",
                    data=csv_data,
                    file_name="resultados_smiles.csv",
                    mime="text/csv",
                )
            else:
                st.warning("Nenhum SMILES válido foi encontrado.")


    elif choice == ("Docking"):
        data_file = st.sidebar.file_uploader("Insert the output from Deep Learning classification",type=['csv'], accept_multiple_files=False)
        st.title("DOCAGEM MOLECULAR NO ALVO DA DENGUE")
 

        if st.button("DOCKING"):
                if data_file is not None:
                    st.write('Verificando moleculas')
        

        # st.subheader("sobre o processamento dos dados")

        # title = st.sidebar.text_input('Insert SMILES here: ')
                    import pandas as pd
                    molecula = pd.read_csv(data_file)
                    try:
                        molecula.drop('Unnamed: 0',axis=1, inplace=True)
                    except:
                        print('')
                    molecula.rename(columns = {'smiles': 'SMILES'}, inplace=True)

                    # anotacao_final.rename(columns={'smiles': 'SMILES'}, inplace=True)
                    raw_html = mols2grid.display(molecula,  subset=["SMILES", 'img'])._repr_html_()
                    components.html(raw_html, width=900, height=300, scrolling=True)
                    # components.html(raw_html)

                    # molecula = molecula.loc[molecula['results']=='Respiratory/Drug']

                    df = molecula

                    # df.rename(columns = {'index': 'index_base1'}, inplace=True)
                    df.rename(columns = {'SMILES': 'smiles'}, inplace=True)
                    # st.dataframe(df)

                    df.reset_index(inplace=True)
                    df.rename(columns = {'index': 'id'}, inplace=True)
                    # st.dataframe(df)

                    df.rename(columns = {'index': 'index_base'}, inplace=True)
                    # df.rename(columns = {'Unnamed: 0', 'index'}, inplace=True)
                    # df.reset_index(inplace=True)

    
                    import os
                    # import git
                    import tempfile

                    import os

                    # Obtenha o caminho para a área de trabalho
                    desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')

                    # Nome da nova pasta a ser criada
                    nova_pasta_nome = "c2tropic-main"

                    # Caminho completo da nova pasta
                    nova_pasta_path = os.path.join(desktop_path, nova_pasta_nome)

                    # Verifica se a pasta já existe antes de criar
                    if not os.path.exists(nova_pasta_path):
                        os.makedirs(nova_pasta_path, exist_ok=True)
                        # st.write(f"A pasta '{nova_pasta_nome}' foi criada na área de trabalho.")
                    else:
                        st.write('Convertendo molécula de texto pra imagem tridimensional')
                        
                        
                    # Cria uma pasta temporária
                    dir_temporaria = nova_pasta_path

                    # st.markdown(f"## ESTE É a pasta: `{dir_temporaria}`")

                    # print(f"Pasta temporária criada em: {dir_temporaria}")

                    # Cria uma pasta dentro da pasta temporária
                    nova_pasta = "runs2"
                    nova_pasta_path = os.path.join(dir_temporaria, nova_pasta)
                    os.makedirs(nova_pasta_path, exist_ok=True)

                    # print(f"Pasta '{nova_pasta}' criada dentro da pasta temporária.")
                    # print(f"Caminho completo da nova pasta: {nova_pasta_path}")


                    # Cria uma pasta dentro da pasta temporária
                    nova_pasta = "teste_teste"
                    nova_pasta_path = os.path.join(dir_temporaria, nova_pasta)
                    os.makedirs(nova_pasta_path, exist_ok=True)

                    
                    df.rename(columns = {'SMILES': 'smiles'}, inplace=True)
                

                    df.rename(columns = {'index': 'index_base'}, inplace=True)

                    ids = []
                    for i in df['id']:
                        a = str(i)
                        ids.append(a)

                    df['id'] = ids

                    # st.markdown ('# antes do mols')

                    # st.dataframe(df)

                    mols = []
                    from rdkit import Chem
                    from rdkit.Chem import AllChem

                    for _, row in df.iterrows():
                        print(row)
                        try:
                            m = Chem.MolFromSmiles(row.smiles)

                            m = Chem.AddHs(m)

                            AllChem.EmbedMolecule(m, AllChem.ETKDG())
                            minimize_status = AllChem.UFFOptimizeMolecule(m, 2000)

                            if not minimize_status == 0:
                                print(f"Failed to minimize_compound'{row['name']}")

                            AllChem.ComputeGasteigerCharges(m)

                            mols.append(m)  
                        except:
                            mols.append(0)   
                            
                    import pandas as pd
                    from rdkit import Chem
                    from rdkit.Chem import AllChem

                    try:
                        df.drop('Unnamed: 0', axis=1, inplace=True)
                    except:
                        print('Nao ha a coluna')

                    # df['index'] = df['index'].astype(str)
                    df['id'] = df['id'].astype(str)

                    st.write(mols)

                    import pandas as pd
                    from rdkit import Chem
                    from rdkit.Chem import AllChem

                    try:
                        df.drop('Unnamed: 0', axis=1, inplace=True)
                    except:
                        print('Nao ha a coluna')

                    # df['index'] = df['index'].astype(str)
                    df['id'] = df['id'].astype(str)

                    # st.markdown('DATAFRAME antes do SDF')
                    # st.write(df)

                    sdf_file_path = os.path.join(dir_temporaria, 'lig_dataset_novo.sdf')

                    # st.markdown(f"## ESTE É O SDF: `{sdf_file_path}`")

                    # Cria o escritor SDF
                    sdf_writer = Chem.SDWriter(sdf_file_path)

                    import pandas as pd
                    from rdkit import Chem
                    from rdkit.Chem import AllChem

                    import numpy as np
                    import os
    
                    pdbqt_folder = os.path.join(dir_temporaria, "teste_teste_NEW_CHEIC")
                    if not os.path.exists(pdbqt_folder):
                        os.makedirs(pdbqt_folder)
                    # sdf_writer = Chem.SDWriter('lig_dataset_novo.sdf')

                    lig_properties = df.columns.to_list()

                    # st.markdown('lig properties')
                    # st.write(lig_properties)

                    for i, mol in enumerate(mols):
                        data_ref = df.iloc[i]
                        mol.SetProp('index', '%s' % i)
                        mol.SetProp('_Name', str(data_ref['id']))  # Convert to string using str()
                        for p in lig_properties:
                            mol.SetProp(f"+{p}", str(data_ref[p]))  # Convert to string using str()
                        sdf_writer.write(mol)
                    sdf_writer.close()

                    import os
                    # import streamlit as st
                    from openbabel import pybel

                    # Título da aplicação
                    st.title("Sua molécula está sendo testada contra dengue")

                    # Caminho do arquivo
                    file_path = os.path.join(os.path.expanduser("~"), "Desktop", "c2tropic-main", "lig_dataset_novo.sdf")

                    # Exibe o diretório atual no Streamlit (para depuração)
                    # st.write(f"Diretório atual: {os.getcwd()}")

                    # Verifica se o arquivo existe
                    if os.path.exists(file_path):
                        st.success(f"O arquivo '{file_path}' foi encontrado!")
                        
                        # Lê e exibe as moléculas no arquivo SDF
                        # st.write("Moléculas no arquivo:")
                        for mol in pybel.readfile('sdf', file_path):
                            st.text(mol)
                    else:
                        st.error(f"Erro: O arquivo '{file_path}' não foi encontrado.")
                        st.info("Certifique-se de que o arquivo esteja no diretório correto ou forneça o caminho absoluto.")



                    from openbabel import pybel
                    import os
                    # os.makedirs(out_dir_lig, exist_ok = True)
                    import tempfile

                    for mol in pybel.readfile('sdf', file_path ):
                        mol.write('pdbqt', 'teste_teste_NEW_CHEIC/%s.pdbqt' %mol.data['index'], overwrite=True)

                    # Caminho completo do arquivo SDF na pasta temporária
                    sdf_file_path = os.path.join(dir_temporaria, 'lig_dataset_novo.sdf')
                    print(sdf_file_path)

                    # Lê os dados do arquivo SDF usando Pybel e escreve em formato PDBQT
                    for mol in pybel.readfile('sdf', sdf_file_path):
                        pdbqt_file_path = os.path.join(pdbqt_folder, '%s.pdbqt' % mol.data['index'])
                        print(pdbqt_file_path)
                        mol.write('pdbqt', pdbqt_file_path, overwrite=True)
                        
                    # RECEPTOR = ['7P2G', '4DD8', '1NC6', '6VVU']
                    RECEPTOR = ['1L9K'] #DENGUE

                    import os
                    df_docagem =[]
                    for i in  RECEPTOR:
                        WORK_DIR = dir_temporaria
                        # st.markdown('dir temporaria')
                        # st.write(dir_temporaria)

                        # LIG_DIR = os.path.join(WORK_DIR, 'teste_teste') #que será temporária
                        LIG_DIR = os.path.join(WORK_DIR, 'teste_teste_NEW_CHEIC')
                        RECEPTOR_DIR = os.path.join(WORK_DIR, 'receptors', i)
                        OUT_DIR = os.path.join(WORK_DIR, "runs2", i)

                        # Verifica se o diretório 'runs2' já existe, caso contrário, cria-o
                        pdbqt_folder = os.path.join(dir_temporaria, "runs2")

                        # Verifica se a pasta de destino para os arquivos PDBQT existe e cria se não existir
                        if not os.path.exists(pdbqt_folder):
                            os.makedirs(pdbqt_folder)
                        
                        
                    #     if not os.path.exists('./runs2/'):
                    #         os.mkdir('./runs2/')

                        # Verifica se o diretório específico para o receptor já existe, caso contrário, cria-o
                        if not os.path.exists(OUT_DIR):
                            os.mkdir(OUT_DIR)

                        ligands = []

                        # Loop para encontrar os arquivos de ligantes
                        for file in os.listdir(LIG_DIR):
                            if os.path.isfile(os.path.join(LIG_DIR, file)) and file.endswith('.pdbqt'):
                                ligands.append(file)
                        
                        
                        
                        prepared_lig_dirs = []
                        from shutil import copy2
                        # Loop para copiar arquivos para o diretório do receptor
                        for lig in ligands:
                            lig_filename = os.path.splitext(lig)[0]
                            out_dir_lig = os.path.join(OUT_DIR, lig_filename)
                            os.makedirs(out_dir_lig, exist_ok=True)

                            copy2(os.path.join(RECEPTOR_DIR, f"{i}final.pdbqt"), out_dir_lig)
                            copy2(os.path.join(RECEPTOR_DIR, f"{i}config.txt"), out_dir_lig)
                            copy2(os.path.join(LIG_DIR, lig), out_dir_lig)

                            prepared_lig_dirs.append(out_dir_lig)
                            
                        import shlex, subprocess
                        from datetime import datetime

                        # st.write('PREPARED LIG DIRS')
                        # st.write(prepared_lig_dirs)

                        # output_logs = ""

                        import subprocess
                        from datetime import datetime
                        import os
                        import streamlit as st

                        # Assuming you have the list prepared_lig_dirs

                        output_logs = ""

                        for j in prepared_lig_dirs:
                            # st.write(f"\n[STARTRUN] {datetime.now()} OUTDIR {j}\n[STARTLOG]\n")

                            ligand = f"{os.path.basename(j)}.pdbqt"
                            # st.write(ligand)

                            vina_exe_path = os.path.join(dir_temporaria, "Vina", "vina.exe")
                            # st.write('vina')
                            # st.write(vina_exe_path)

                            args = [vina_exe_path, '--receptor', f"{i}final.pdbqt", '--config', 
                                    f"{i}config.txt", '--ligand', ligand, '--log', 'results.txt']
                            # st.write('args da docagem')
                            # st.write(args)

                            process = subprocess.Popen(
                                args,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                cwd=os.path.join(j)
                            )

                            # Display the output dynamically
                            st_output = st.empty()
                            while process.poll() is None:
                                output = process.stdout.readline().decode("utf-8")
                                st_output.write(output)

                            output, error = process.communicate()

                            if error:
                                # st.write("Error:", error.decode("utf-8"))
                                output_logs = output_logs + error.decode("utf-8")
                            else:
                                # st.write("Output:", output.decode("utf-8"))
                                output_logs = output_logs + output.decode("utf-8")

                            output_logs = output_logs + f"\n[ENDLOG]\n[ENDRUN] {datetime.now()}\n+++++++++++++\n"



                        
                        WORK_DIR = os.path.join(dir_temporaria, 'runs2')
                        WORK_DIR = os.path.join(WORK_DIR, i)
                        ligands = []
                        for file in os.listdir(WORK_DIR):
                            ligands.append(file)
                        #         print('Este sao os ligantes')
                        #         print(ligands)
                        #         print('+++++++++++++++++++++')

                        prepared_lig_dirs = []
                        for b in ligands:
                            #lig_filename = os.path.splitext(lig)[0]
                            out_dir_lig = os.path.join(WORK_DIR, b)
                            prepared_lig_dirs.append(out_dir_lig)
                        #         print('Este sao os prepared_lig_dirs')
                        #        0 print(prepared_lig_dirs)
                        #         print('+++++++++++++++++++++')

                        endereco = []
                        for q in prepared_lig_dirs:
                            a = q + '\\results.txt'
                        #             print(a)
                            endereco.append(a)
                            
                        df_docagem1 = []

                        import pandas as pd
                        for t in endereco:

                            data = pd.read_csv(t, header = None, on_bad_lines='skip')

                            if len(data) == 12:
                                df_docagem1.append(0)

                            else:

                                best_pose = pd.DataFrame(data[0][20].split('      ')).T
                                try:
                                    a=best_pose.iat[0, 1].strip(" ")
                                except:
                                    a=0
                        #             a=best_pose.iat[0, 1].strip(" ")


                                df_docagem1.append(a)
                        df_docagem.append(df_docagem1)

                        
                    docking = pd.DataFrame(df_docagem).T

                    for i, j in enumerate(RECEPTOR):
                                # print(i, j)
                                #a = j
                        docking.rename(columns = {i: j}, inplace=True)

                    # docking.reset_index(inplace=True)
                    # st.dataframe(docking)

                    docking['index'] = ligands
                    docking['index'] = docking['index'].astype(int)
                                                
                    df.reset_index(inplace=True)
                    anotacoes_com_docagem = df.merge(docking, how='inner', on='index')
                    anotacoes_com_docagem


                st.write('Results with poses:')
                st.dataframe(anotacoes_com_docagem)


                @st.cache_data
                def convert_df(df):
                    # IMPORTANT: Cache the conversion to prevent computation on every rerun
                    return df.to_csv(index=False).encode('utf-8')

                csv = convert_df(anotacoes_com_docagem)

                st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='amostras_machine_learning_docking.csv',
                mime='text/csv',
                )


                
    else:
        st.subheader("About")
        # st.dataframe(amostras)
        
        image = Image.open('rafael.png')

        st.image(image, use_container_width=True)
        # st.info("Built with Streamlit")
        st.info("Rafael Vieira")
        



if __name__ == '__main__':
    main()
