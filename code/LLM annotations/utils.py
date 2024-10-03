from tqdm import tqdm


def generate_prompt(query, title, doc):
    prompt = f'''You're an annotator chosen for a task of annotating the documents retrieved in response to the queries about controversial queries that we issued to the search engines, Bing and Google. 
    The documents you will annotate have been chosen from the top-10 search results retrieved from these search engines. 
    You're allowed to read the query and the corresponding document, then annotate the document with respect to the given query first as relevant or not-relevant, then if the document is relevant, you should annotate the document as pro, neutral, or against.

    Pro: when the document is in favor of the controversial topic. The document describes more the pro aspects of the topic; 

    Neutral: when the document does not support or help either side of the controversial topic. The document provides an impartial (fair) description of the pros and cons of the subject;

    Against: when the document is against the controversial topic. The document describes more the cons aspects of the topic;
    
    Not-relevant: when the document is irrelevant regarding the controversial topic;

    QUERY:

    {query}

    DOCUMENT TITLE:

    {title}

    DOCUMENT:

    {doc}
    '''
    return prompt


def annotate_dataframe(df, generator, col_name, output_file=None):
    """
    Annotates the DataFrame by generating prompts and corresponding generated text.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing 'Query', 'docTitle', and 'doc' columns.
    generator (function): The text generation function to apply to each prompt.
    output_file (str): Optional. The file path to save the annotated DataFrame as a CSV.
    
    Returns:
    pd.DataFrame: The annotated DataFrame with 'Prompt' and 'GeneratedText' columns added.
    """
    
    # Initialize tqdm progress bar for applying the function to rows
    tqdm.pandas(desc="Annotating DataFrame")
    
    # Generate prompts based on the 'Query', 'docTitle', and 'doc' columns
    df['Prompt'] = df.progress_apply(lambda row: generate_prompt(row['Query'], row['docTitle'], row['doc']), axis=1)
    df['Prompt'] = df['Prompt'].str.replace('\n', ' ')
    # Apply the text generation function to each prompt and store the results
    df[col_name] = df['Prompt'].progress_apply(generator)
    
    # Optionally save the DataFrame to a CSV file
    if output_file:
        df.to_csv(output_file, index=False)
    
    return df